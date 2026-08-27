Привести структуру проекта к виду

my_project/                  # Корень проекта (можно называть как угодно)
├── src/                     # Исходный код (главная папка)
│   └── my_app/              # Название вашего пакета (Python package)
│       ├── __init__.py
│       ├── main.py          # Точка входа / фасад
│       ├── models.py        # Модели данных (БД, Pydantic, dataclasses)
│       ├── services.py      # Бизнес-логика
│       ├── config.py        # Настройки (pydantic-settings)
│       └── utils.py         # Вспомогательные функции
├── tests/                   # Тесты (зеркалируют структуру src/)
│   ├── __init__.py
│   ├── conftest.py          # Фикстуры pytest
│   ├── unit/
│   │   └── test_services.py
│   └── integration/
│       └── test_api.py
├── scripts/                 # Служебные скрипты (миграции БД, заполнение данными)
├── pyproject.toml           # Главный файл конфигурации (замена setup.py)
├── requirements.txt         # (опционально) или только зависимости в pyproject.toml
├── .env                     # Переменные окружения (не в Git!)
├── .gitignore
├── Makefile / tasks.py      # Автоматизация (запуск тестов, линтеров)
└── README.md