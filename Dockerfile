# Образ песочницы для исполнения команд агента (см. openspec/changes/add-docker-sandbox).
# Alpine + минимальный набор утилит скиллов: curl (wttr.in, habr),
# python3 со stdlib и pip для установок `pip install --user` в /work.
FROM alpine:3.20

RUN apk add --no-cache curl python3 py3-pip

# Записываемый рабочий каталог для непривилегированного пользователя (uid 1000)
# и HOME для pip --user / git / __pycache__ — без этого -u 1000 ломает установки.
RUN mkdir /work && chown 1000:1000 /work
ENV HOME=/work
# PEP 668: образ песочницы эфемерный, изоляция системных пакетов не нужна —
# разрешаем модели `pip install --user` без флага --break-system-packages.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /work
