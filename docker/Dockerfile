FROM python:3.11.6

WORKDIR /app

COPY . /app

RUN pip install poetry

RUN poetry install

CMD [ "poetry", "run", "python", "-m", "src.app.bot" ]
