## 📊 Auto Status 1-Pager

Инструмент для **автоматической генерации отчётов о статусе проектов**.
Создаёт файлы в форматах **Markdown** и **HTML** с графиками и таблицами.

---

### 🚀 Как запустить

1. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Запустите генератор:

   ```bash
   python src/report_generator.py
   ```

После запуска появятся файлы:

* `data/project_status.csv` — данные (создаются автоматически при первом запуске);
* `status_report.md` — отчёт в Markdown;
* `status_report.html` — отчёт в HTML;
* `figs/` — изображения графиков.

---

### 📂 Структура проекта

```
auto_status_1pager/
├── notebooks/status_report_generator.ipynb
├── src/report_generator.py
    ├── data/project_status.csv
    ├── figs/
    ├── status_report.md
    ├── status_report.html
└── requirements.txt
```

---

### 📈 Что содержит отчёт

* Средний прогресс по проектам
* Распределение задач по статусам
* Приоритеты и исполнители
* Дата генерации отчёта

---

### 👤 Автор

**Артём**, факультет информационных технологий.


