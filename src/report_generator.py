#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
from datetime import datetime
from collections import Counter
import base64
from io import BytesIO
import html as html_mod

import numpy as np
import pandas as pd
import matplotlib

# Неинтерактивный бэкенд (работает на серверах / без GUI)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# ---- Константы и утилиты ----
RUS_MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
}


def format_date_ru(dt):
    """Форматирует дату в виде '14 февраля 2024' или возвращает пустую строку."""
    if pd.isna(dt) or dt is None:
        return ""
    if not isinstance(dt, (datetime, pd.Timestamp)):
        try:
            dt = pd.to_datetime(dt, errors="coerce")
            if pd.isna(dt):
                return ""
        except Exception:
            return str(dt)
    return f"{dt.day} {RUS_MONTHS.get(dt.month, '')} {dt.year}"


def create_sample_csv(path: str):
    """Создаёт пример CSV с набором записей, если файла нет."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sample = (
        "project,task,assigned_to,status,progress,blockers,due_date,priority\n"
        "Проект Альфа,Дизайн интерфейса,Иван,In Progress,75%,,2024-02-15,High\n"
        "Проект Альфа,Реализация фронтенда,Алиса,In Progress,50%,API не готов,2024-02-20,Medium\n"
        "Проект Альфа,Интеграция платежей,Олег,Blocked,20%,Нет доступа к платёжному провайдеру,2024-03-01,High\n"
        "Проект Альфа,Тестирование UI,Елена,Not Started,0%,,2024-03-10,Low\n"
        "Проект Бета,Миграция данных,Михаил,Blocked,30%,Ожидание от клиента,2024-02-28,High\n"
        "Проект Бета,Написать скрипты миграции,Борис,In Progress,60%,,2024-02-25,High\n"
        "Проект Бета,Проверка целостности данных,Ольга,In Progress,40%,Нужен доступ к staging,2024-03-05,Medium\n"
        "Проект Гамма,Разработка API,Андрей,In Progress,55%,,2024-04-01,High\n"
        "Проект Гамма,Документация API,Мария,Not Started,0%,,2024-04-10,Low\n"
        "Проект Гамма,Автотесты,Екатерина,Not Started,0%,,2024-04-15,Medium\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"Создан пример данных: {path}")


# ---- Загрузка и предобработка данных ----
def load_data(csv_path: str) -> pd.DataFrame:
    """Загружает CSV, при отсутствии создаёт пример. Возвращает предобработанный DataFrame."""
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден — создаю пример...")
        create_sample_csv(csv_path)

    df = pd.read_csv(csv_path, dtype=str)

    # Ожидаемые колонки — если их нет, добавляем пустые
    expected = ["project", "task", "assigned_to", "status", "progress", "blockers", "due_date", "priority"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""

    # Очищаем пробелы только в строковых колонках (без applymap)
    str_cols = df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Парсинг progress (например "75%")
    def parse_progress(x):
        try:
            if pd.isna(x) or str(x).strip() == "":
                return 0.0
            s = str(x).strip().replace(",", ".")
            if s.endswith("%"):
                s = s[:-1]
            return float(s)
        except Exception:
            return 0.0

    df["progress_pct"] = df["progress"].apply(parse_progress)

    # Парсинг дат дедлайнов
    df["due_date_parsed"] = pd.to_datetime(df["due_date"], errors="coerce")

    # Нормализация статусов
    def normalize_status(s):
        if not isinstance(s, str):
            return "Неизвестно"
        st = s.strip().lower()
        if st in ["completed", "done", "выполнено"]:
            return "Выполнено"
        if st in ["in progress", "inprogress", "в работе"]:
            return "В работе"
        if st in ["blocked", "заблокировано"]:
            return "Заблокировано"
        if st in ["not started", "notstarted", "не начато"]:
            return "Не начато"
        return s.strip().capitalize()

    df["status_norm"] = df["status"].apply(normalize_status)

    # Нормализация приоритетов
    def normalize_priority(p):
        if not isinstance(p, str) or p.strip() == "":
            return "Не указан"
        p_low = p.strip().lower()
        if p_low in ["high"]:
            return "Высокий"
        if p_low in ["medium", "med"]:
            return "Средний"
        if p_low in ["low"]:
            return "Низкий"
        return p.strip().capitalize()

    df["priority_norm"] = df["priority"].apply(normalize_priority)

    # Исполнители
    df["assigned_to"] = df["assigned_to"].fillna("").replace("", "Не назначен")

    return df


# ---- Вычисление статистик ----
def compute_statistics(df: pd.DataFrame) -> dict:
    """Вычисляет расширенные метрики и возвращает словарь со всеми данными."""
    total_tasks = len(df)
    status_counts = df["status_norm"].value_counts().to_dict()
    priority_counts = df["priority_norm"].value_counts().to_dict()
    assignee_counts = df["assigned_to"].value_counts().to_dict()

    completed = status_counts.get("Выполнено", 0)
    in_progress = status_counts.get("В работе", 0)
    blocked = status_counts.get("Заблокировано", 0)
    not_started = status_counts.get("Не начато", 0)

    project_progress = df.groupby("project")["progress_pct"].mean().round(2)
    overall_progress = float(df["progress_pct"].mean().round(2)) if total_tasks > 0 else 0.0

    now = pd.Timestamp(datetime.now())
    overdue_df = df[pd.notnull(df["due_date_parsed"]) & (df["due_date_parsed"] < now)].copy()
    if not overdue_df.empty:
        overdue_df["days_overdue"] = (now - overdue_df["due_date_parsed"]).dt.days
    else:
        overdue_df["days_overdue"] = []

    blockers_texts = df["blockers"].fillna("").astype(str)
    blockers_list = [b.strip() for b in blockers_texts if isinstance(b, str) and b.strip() != ""]
    blockers_counter = Counter(blockers_list)
    top_blockers = blockers_counter.most_common(10)

    upcoming = df[pd.notnull(df["due_date_parsed"])].sort_values("due_date_parsed").head(10)

    stats = {
        "total_tasks": int(total_tasks),
        "status_counts": status_counts,
        "priority_counts": priority_counts,
        "assignee_counts": assignee_counts,
        "completed": int(completed),
        "in_progress": int(in_progress),
        "blocked": int(blocked),
        "not_started": int(not_started),
        "project_progress": project_progress,
        "overall_progress": overall_progress,
        "overdue_df": overdue_df,
        "top_blockers": top_blockers,
        "upcoming": upcoming,
    }
    return stats


# ---- Генерация графиков (сохранение в figs/) ----
def generate_plots(df: pd.DataFrame, out_dir: str) -> dict:
    """Генерирует PNG-графики и возвращает словарь путей."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    figs = {}

    # 1. Круговая диаграмма по статусам
    status_series = df["status_norm"].fillna("Неизвестно").value_counts()
    plt.figure(figsize=(6, 6))
    status_series.plot.pie(autopct="%1.1f%%", ylabel="", title="Распределение задач по статусу")
    pie_path = os.path.join(out_dir, "status_distribution_pie.png")
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()
    figs["status_pie"] = pie_path

    # 2. Столбчатая диаграмма: средний прогресс по проектам
    proj_progress = df.groupby("project")["progress_pct"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=proj_progress.values, y=proj_progress.index)
    plt.xlabel("Средний прогресс (%)")
    plt.ylabel("")
    plt.title("Средний прогресс по проектам")
    bar_path = os.path.join(out_dir, "progress_by_project_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    figs["progress_bar"] = bar_path

    # 3. Круговая диаграмма по приоритетам
    prio = df["priority_norm"].fillna("Не указан").value_counts()
    plt.figure(figsize=(6, 6))
    prio.plot.pie(autopct="%1.1f%%", ylabel="", title="Распределение по приоритетам")
    prio_path = os.path.join(out_dir, "priority_distribution_pie.png")
    plt.tight_layout()
    plt.savefig(prio_path)
    plt.close()
    figs["priority_pie"] = prio_path

    # 4. Нагрузка по исполнителям (топ-10)
    assignee_counts = df["assigned_to"].fillna("Не назначен").value_counts().head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=assignee_counts.values, y=assignee_counts.index)
    plt.xlabel("Количество задач")
    plt.ylabel("")
    plt.title("Нагрузка по исполнителям (топ-10)")
    assignee_path = os.path.join(out_dir, "assignee_load_bar.png")
    plt.tight_layout()
    plt.savefig(assignee_path)
    plt.close()
    figs["assignee_bar"] = assignee_path

    return figs


# ---- Вспомогательные функции для встраивания изображений в HTML ----
def img_file_to_data_uri(path: str) -> str:
    """Читает PNG и возвращает data-uri (base64)."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---- Генерация Markdown отчёта ----
def generate_markdown_report(df: pd.DataFrame, stats: dict, figs: dict, out_path: str, generation_time: datetime):
    """Генерирует Markdown-отчёт (русский язык)."""
    now = generation_time
    now_str = f"{now.day} {RUS_MONTHS[now.month]} {now.year} {now.strftime('%H:%M:%S')}"

    md_lines = []
    md_lines.append(f"# Отчёт о статусе проектов\n_Дата генерации: {now_str}_\n")

    # Короткая сводка
    md_lines.append("## Короткая сводка\n")
    md_lines.append(f"- **Всего задач:** {stats['total_tasks']}")
    md_lines.append(f"- **Средний прогресс:** {stats['overall_progress']}%")
    md_lines.append(f"- **Выполнено:** {stats['completed']}; **В работе:** {stats['in_progress']}; **Заблокировано:** {stats['blocked']}\n")

    # Рекомендации (короткие)
    md_lines.append("## Рекомендуемые действия (сводка)\n")
    recs = []
    if stats["blocked"] > 0:
        recs.append("- Эскалация по заблокированным задачам: организовать срочный стендап с ответственными и заказчиком.")
    if stats["overall_progress"] < 50:
        recs.append("- Общий прогресс ниже 50% — пересмотреть приоритеты и перераспределить ресурсы.")
    if not stats["overdue_df"].empty:
        recs.append(f"- Есть просроченные задачи ({len(stats['overdue_df'])}) — провести ревью и определить ответственных.")
    if not recs:
        recs.append("- Явных критических проблем не обнаружено. Продолжать в текущем режиме.")
    md_lines.extend(recs)
    md_lines.append("")

    # Статистика по статусам
    md_lines.append("## Статистика по статусам\n")
    md_lines.append("| Статус | Количество |")
    md_lines.append("|---|---:|")
    for s, c in stats["status_counts"].items():
        md_lines.append(f"| {s} | {c} |")
    md_lines.append("")

    # Визуализации
    md_lines.append("## Визуализации\n")
    mapping = {
        "status_pie": "Распределение по статусам",
        "progress_bar": "Средний прогресс по проектам",
        "priority_pie": "Распределение по приоритетам",
        "assignee_bar": "Нагрузка по исполнителям (топ-10)"
    }
    for key, title in mapping.items():
        if key in figs:
            rel = os.path.relpath(figs[key], os.path.dirname(out_path) or ".")
            md_lines.append(f"### {title}\n")
            md_lines.append(f"![{title}]({rel})\n")

    # Ближайшие дедлайны
    md_lines.append("## Ближайшие дедлайны (топ-10)\n")
    md_lines.append("| Проект | Задача | Исполнитель | Дедлайн |")
    md_lines.append("|---|---|---|---:|")
    for _, row in stats["upcoming"].iterrows():
        md_lines.append("| {} | {} | {} | {} |".format(
            row.get("project", ""),
            row.get("task", ""),
            row.get("assigned_to", ""),
            format_date_ru(row.get("due_date_parsed"))
        ))
    md_lines.append("")

    # Просроченные задачи
    md_lines.append("## Просроченные задачи\n")
    if stats["overdue_df"].empty:
        md_lines.append("_Просроченных задач не обнаружено._\n")
    else:
        md_lines.append("| Проект | Задача | Исполнитель | Дедлайн | Просрочено (дней) |")
        md_lines.append("|---|---|---|---:|---:|")
        for _, r in stats["overdue_df"].sort_values("days_overdue", ascending=False).iterrows():
            md_lines.append("| {} | {} | {} | {} | {} |".format(
                r.get("project", ""),
                r.get("task", ""),
                r.get("assigned_to", ""),
                format_date_ru(r.get("due_date_parsed")),
                int(r.get("days_overdue", 0))
            ))
        md_lines.append("")

    # Топ блокеров
    md_lines.append("## Топ причин блокировки (по упоминаниям)\n")
    if stats["top_blockers"]:
        md_lines.append("| Причина блокировки | Количество упоминаний |")
        md_lines.append("|---|---:|")
        for txt, cnt in stats["top_blockers"]:
            md_lines.append(f"| {txt} | {cnt} |")
    else:
        md_lines.append("_Причин блокировок не указано._")
    md_lines.append("")

    # Полная таблица задач
    md_lines.append("## Полная таблица задач\n")
    md_lines.append("| Проект | Задача | Исполнитель | Статус | Прогресс | Блокеры | Дедлайн | Приоритет |")
    md_lines.append("|---|---|---|---|---:|---|---:|---|")
    for _, r in df.iterrows():
        md_lines.append("| {} | {} | {} | {} | {}% | {} | {} | {} |".format(
            r.get("project", ""),
            r.get("task", ""),
            r.get("assigned_to", ""),
            r.get("status_norm", ""),
            int(r.get("progress_pct", 0)),
            (r.get("blockers", "") or ""),
            format_date_ru(r.get("due_date_parsed")),
            r.get("priority_norm", "")
        ))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Markdown-отчёт сформирован: {out_path} (время генерации: {now_str})")


# ---- Генерация HTML отчёта (single-file) ----
def generate_html_report(df: pd.DataFrame, stats: dict, figs: dict, out_path: str, generation_time: datetime):
    """Генерирует самодостаточный HTML с embedded изображениями (data-uri)."""
    now = generation_time
    now_str = f"{now.day} {RUS_MONTHS[now.month]} {now.year} {now.strftime('%H:%M:%S')}"

    imgs = {}
    for key, path in figs.items():
        try:
            imgs[key] = img_file_to_data_uri(path)
        except Exception:
            imgs[key] = ""

    html_lines = []
    html_lines.append("<!doctype html>")
    html_lines.append("<html lang='ru'>")
    html_lines.append("<head>")
    html_lines.append("<meta charset='utf-8'>")
    html_lines.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    # Метатеги против кэширования
    html_lines.append("<meta http-equiv='Cache-Control' content='no-store, no-cache, must-revalidate' />")
    html_lines.append("<meta http-equiv='Pragma' content='no-cache' />")
    html_lines.append("<meta http-equiv='Expires' content='0' />")
    html_lines.append(f"<title>Отчёт о статусе проектов — {html_mod.escape(now_str)}</title>")

    html_lines.append("""<style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #222; background: #f9f9f9; }
      h1,h2,h3 { color: #113; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
      .card { padding: 12px; border: 1px solid #e6e6e6; border-radius: 6px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; }
      table, th, td { border: 1px solid #eee; }
      th, td { padding: 8px; text-align: left; }
      th { background: #fafafa; }
      img { max-width: 100%; height: auto; display:block; margin: 0 auto; }
      .small { font-size: 0.9em; color: #555; }
      .muted { color: #666; font-size: 0.95em; }
    </style>""")
    html_lines.append("</head><body>")

    html_lines.append(f"<h1>Отчёт о статусе проектов</h1>")
    html_lines.append(f"<p class='muted'>Дата генерации: {html_mod.escape(now_str)}</p>")

    # Короткая сводка
    html_lines.append("<div class='card'>")
    html_lines.append("<h2>Короткая сводка</h2>")
    html_lines.append("<p>")
    html_lines.append(f"Всего задач: <strong>{stats['total_tasks']}</strong>.<br>")
    html_lines.append(f"Средний прогресс по всем задачам: <strong>{stats['overall_progress']}%</strong>.<br>")
    html_lines.append(f"Выполнено: <strong>{stats['completed']}</strong>; В работе: <strong>{stats['in_progress']}</strong>; Заблокировано: <strong>{stats['blocked']}</strong>.")
    html_lines.append("</p></div>")

    # Рекомендации
    html_lines.append("<div class='card'><h2>Рекомендуемые действия (сводка)</h2><ul>")
    if stats["blocked"] > 0:
        html_lines.append("<li>Эскалация по заблокированным задачам: организовать срочный стендап с ответственными и заказчиком.</li>")
    if stats["overall_progress"] < 50:
        html_lines.append("<li>Общий прогресс ниже 50% — пересмотреть приоритеты и перераспределить ресурсы.</li>")
    if not stats["overdue_df"].empty:
        html_lines.append(f"<li>Есть просроченные задачи ({len(stats['overdue_df'])}) — провести ревью и определить ответственных.</li>")
    if stats["blocked"] == 0 and stats["overall_progress"] >= 50 and stats["overdue_df"].empty:
        html_lines.append("<li>Явных критических проблем не обнаружено. Продолжать в текущем режиме.</li>")
    html_lines.append("</ul></div>")

    # Статистика таблицами в гриде
    html_lines.append("<div class='grid'>")
    # Статусы
    html_lines.append("<div class='card'><h3>Статистика по статусам</h3><table><tr><th>Статус</th><th>Количество</th></tr>")
    for s, c in stats["status_counts"].items():
        html_lines.append(f"<tr><td>{html_mod.escape(str(s))}</td><td style='text-align:right'>{c}</td></tr>")
    html_lines.append("</table></div>")
    # Приоритеты
    html_lines.append("<div class='card'><h3>Статистика по приоритетам</h3><table><tr><th>Приоритет</th><th>Количество</th></tr>")
    for p, c in stats["priority_counts"].items():
        html_lines.append(f"<tr><td>{html_mod.escape(str(p))}</td><td style='text-align:right'>{c}</td></tr>")
    html_lines.append("</table></div>")
    html_lines.append("</div>")  # grid

    # Визуализации (встраиваем)
    html_lines.append("<h2>Визуализации</h2>")
    mapping_titles = {
        "status_pie": "Распределение по статусам",
        "progress_bar": "Средний прогресс по проектам",
        "priority_pie": "Распределение по приоритетам",
        "assignee_bar": "Нагрузка по исполнителям (топ-10)"
    }
    for key, title in mapping_titles.items():
        if key in imgs and imgs[key]:
            html_lines.append("<div class='card'>")
            html_lines.append(f"<h3>{html_mod.escape(title)}</h3>")
            html_lines.append(f"<img src='{imgs[key]}' alt='{html_mod.escape(title)}'/>")
            html_lines.append("</div>")

    # Ближайшие дедлайны
    html_lines.append("<div class='card'><h2>Ближайшие дедлайны (топ-10)</h2>")
    html_lines.append("<table><tr><th>Проект</th><th>Задача</th><th>Исполнитель</th><th>Дедлайн</th></tr>")
    for _, row in stats["upcoming"].iterrows():
        html_lines.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html_mod.escape(str(row.get("project", ""))),
            html_mod.escape(str(row.get("task", ""))),
            html_mod.escape(str(row.get("assigned_to", ""))),
            html_mod.escape(format_date_ru(row.get("due_date_parsed")))
        ))
    html_lines.append("</table></div>")

    # Просроченные задачи
    html_lines.append("<div class='card'><h2>Просроченные задачи</h2>")
    if stats["overdue_df"].empty:
        html_lines.append("<p>Просроченных задач не обнаружено.</p>")
    else:
        html_lines.append("<table><tr><th>Проект</th><th>Задача</th><th>Исполнитель</th><th>Дедлайн</th><th>Просрочено (дн)</th></tr>")
        for _, r in stats["overdue_df"].sort_values("days_overdue", ascending=False).iterrows():
            html_lines.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td style='text-align:right'>{}</td></tr>".format(
                html_mod.escape(str(r.get("project", ""))),
                html_mod.escape(str(r.get("task", ""))),
                html_mod.escape(str(r.get("assigned_to", ""))),
                html_mod.escape(format_date_ru(r.get("due_date_parsed"))),
                int(r.get("days_overdue", 0))
            ))
        html_lines.append("</table>")
    html_lines.append("</div>")

    # Топ блокеров
    html_lines.append("<div class='card'><h2>Топ причин блокировки</h2>")
    if stats["top_blockers"]:
        html_lines.append("<table><tr><th>Причина</th><th>Упоминаний</th></tr>")
        for txt, cnt in stats["top_blockers"]:
            html_lines.append("<tr><td>{}</td><td style='text-align:right'>{}</td></tr>".format(html_mod.escape(txt), cnt))
        html_lines.append("</table>")
    else:
        html_lines.append("<p>Причин блокировок не указано.</p>")
    html_lines.append("</div>")

    # Полная таблица задач
    html_lines.append("<div class='card'><h2>Полная таблица задач</h2>")
    html_lines.append("<table><tr><th>Проект</th><th>Задача</th><th>Исполнитель</th><th>Статус</th><th>Прогресс</th><th>Блокеры</th><th>Дедлайн</th><th>Приоритет</th></tr>")
    for _, r in df.iterrows():
        html_lines.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td style='text-align:right'>{}%</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html_mod.escape(str(r.get("project", ""))),
            html_mod.escape(str(r.get("task", ""))),
            html_mod.escape(str(r.get("assigned_to", ""))),
            html_mod.escape(str(r.get("status_norm", ""))),
            int(r.get("progress_pct", 0)),
            html_mod.escape(str(r.get("blockers", "") or "")),
            html_mod.escape(format_date_ru(r.get("due_date_parsed"))),
            html_mod.escape(str(r.get("priority_norm", "")))
        ))
    html_lines.append("</table></div>")

    html_lines.append(f"<footer class='muted small' style='margin-top:20px;'>Отчёт сгенерирован {html_mod.escape(now_str)}</footer>")
    html_lines.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"HTML-отчёт сформирован: {out_path} (время генерации: {now_str})")


# ---- main ----
def main():
    parser = argparse.ArgumentParser(description="Генератор отчётов о статусе проектов (русский язык)")
    parser.add_argument("--csv", default="data/project_status.csv", help="Путь к CSV-файлу")
    parser.add_argument("--out", default=None, help="Путь к Markdown-отчёту (по умолчанию status_report_YYYYmmdd_HHMMSS.md)")
    parser.add_argument("--html", default=None, help="Путь к HTML-отчёту (по умолчанию status_report_YYYYmmdd_HHMMSS.html)")
    parser.add_argument("--figs", default="figs", help="Каталог для временных графиков")
    args = parser.parse_args()

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    out_md = args.out if args.out else f"status_report_{ts}.md"
    out_html = args.html if args.html else f"status_report_{ts}.html"

    print(f"Генерация отчёта: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV: {args.csv}")
    print(f"Markdown -> {out_md}")
    print(f"HTML -> {out_html}")

    df = load_data(args.csv)
    stats = compute_statistics(df)
    figs = generate_plots(df, args.figs)

    generate_markdown_report(df, stats, figs, out_md, now)
    generate_html_report(df, stats, figs, out_html, now)


if __name__ == "__main__":
    main()

