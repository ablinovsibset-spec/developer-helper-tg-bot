# sample_project

Учебный проект для файловых сценариев benchmark-харнесса (add-token-audit):
маленький python-проект с модулями и unittest-тестами. Доставляется в
песочницу в `/work/sample_project` (design D5) и служит материалом для
задач агенту на чтение файлов и запуск тестов.

Запуск тестов из корня проекта:

    python3 -m unittest discover -s tests -t .

В проекте есть известный баг: `calculate_total` не применяет скидку при
сумме выше `DISCOUNT_THRESHOLD`, из-за чего `test_total_applies_discount`
падает — падающие и проходящие тесты нужны сценарию одновременно.
