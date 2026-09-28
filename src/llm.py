import argparse
import asyncio
import json
import os
import time
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from eval import load_bench, make_holdout
from lexical import service_kind
from retrieval import CACHE, topk
from signals import normalize_query

ServiceKind = Literal[
    "Обучение, курсы",
    "Красота, здоровье",
    "Ремонт и отделка",
    "Строительство",
    "Автосервис, аренда",
    "Деловые услуги",
    "Праздники, мероприятия",
    "Оборудование, производство",
    "Ремонт и обслуживание техники",
    "Сад, благоустройство",
    "Грузоперевозки",
    "Вывоз мусора и вторсырья",
    "Фото- и видеосъёмка",
    "Пассажирские перевозки",
    "Другое",
    "Искусство",
    "Компьютерная помощь",
    "Бытовые услуги",
    "Уход за животными",
    "Доставка еды и продуктов",
    "Уборка",
    "Услуги эвакуатора",
    "Монтаж и установка техники",
    "Няни, сиделки",
    "Грузчики, складские услуги",
    "Охрана, безопасность",
]


class Reasoned(BaseModel):
    reason_steps: list[str] = Field(
        min_length=3,
        max_length=7,
        description=(
            "Пошаговое рассуждение до ответа, от 3 до 7 коротких шагов: что ищет пользователь, какие слова "
            "запроса ключевые, есть ли в нём опечатки, сокращения или сленг, к какой сфере услуг он "
            "относится и что подсказывают похожие запросы из истории."
        ),
    )


class Rewrite(Reasoned):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Поисковый запрос пользователя, переписанный языком объявлений исполнителей на Авито."
            )
        }
    )
    answer: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Запрос так, как исполнители написали бы заголовок своего объявления: без опечаток, с раскрытыми "
            "сокращениями и сленгом, плюс 2–4 близких по смыслу слова. Без города и цены, не длиннее 15 слов."
        ),
    )


class Kinds(Reasoned):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Виды услуг, к которым относятся объявления, которые ищет пользователь на Авито."
        }
    )
    answer: list[ServiceKind] = Field(
        min_length=1,
        max_length=3,
        description="От 1 до 3 видов услуг из списка допустимых значений, самый вероятный первым.",
    )


TASKS: dict[str, type[Reasoned]] = {"rewrite": Rewrite, "kinds": Kinds}

EXAMPLES = 5
RETRIES = 3
MAX_TOKENS = 600


def system_prompt(task: str) -> str:
    schema = json.dumps(TASKS[task].model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Ты помогаешь поиску услуг на Авито. Задача описана в JSON-схеме ответа ниже: что нужно сделать, "
        "сказано в description схемы, смысл каждого поля — в его description. "
        "Сначала заполни reason_steps, затем answer. Отвечай только JSON строго по схеме:\n"
        f"{schema}"
    )


def user_prompt(query: str, examples: list[tuple[str, str, str]]) -> str:
    shots = "\n".join(
        f"- «{q}» → «{title}» (вид услуги: {kind or 'не указан'})" for q, title, kind in examples
    )
    header = "Похожие запросы из истории поиска и объявления, которые по ним выбрали:"
    return f"{header}\n{shots}\n\nЗапрос: «{query}»"


def response_format(task: str) -> ResponseFormatJSONSchema:
    return {
        "type": "json_schema",
        "json_schema": {"name": task, "schema": TASKS[task].model_json_schema(), "strict": True},
    }


def few_shot(texts: list[str], rest: pd.DataFrame) -> list[list[tuple[str, str, str]]]:
    table = (
        rest.assign(text=normalize_query(rest["search_query"]))
        .groupby("text")
        .agg(title=("item_title_raw", "first"), params=("item_infm_params_text", "first"))
    )
    texts_known = [str(t) for t in table.index]
    titles = [str(t) for t in table["title"]]
    kinds = [service_kind(p) for p in table["params"]]
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32
    )
    TT = vec.fit_transform(texts_known).T.tocsr()
    Q = vec.transform(texts)
    out: list[list[tuple[str, str, str]]] = []
    for i in range(0, len(texts), 512):
        idx, _ = topk((Q[i : i + 512] @ TT).toarray(), EXAMPLES)
        out += [[(texts_known[j], titles[j], kinds[j]) for j in row] for row in idx]
    return out


def client() -> tuple[AsyncOpenAI, str]:
    load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise SystemExit(
            "нужна переменная LLM_BASE_URL (адрес llama-server, например http://localhost:8080/v1)"
        )
    api = AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ.get("LLM_API_KEY", "local"),
        http_client=DefaultAsyncHttpxClient(trust_env=False),
    )
    return api, os.environ.get("LLM_MODEL", "local")


async def ask(
    api: AsyncOpenAI, model: str, task: str, query: str, examples: list[tuple[str, str, str]]
) -> Reasoned:
    error: Exception | None = None
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt(task)},
        {"role": "user", "content": user_prompt(query, examples)},
    ]
    for _ in range(RETRIES):
        response = await api.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format(task),
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        try:
            return TASKS[task].model_validate_json(response.choices[0].message.content or "")
        except ValidationError as e:
            error = e
    assert error is not None
    raise error


def cache_path(task: str) -> str:
    return str(CACHE / f"llm_{task}.jsonl")


def answers(task: str) -> dict[str, object]:
    path = cache_path(task)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["query"]: row["answer"] for row in rows}


async def run(task: str, texts: list[str], examples: list[list[tuple[str, str, str]]]) -> None:
    api, model = client()
    done = set(answers(task))
    todo = [(t, ex) for t, ex in zip(texts, examples, strict=True) if t not in done]
    todo = list(dict((t, (t, ex)) for t, ex in todo).values())
    semaphore = asyncio.Semaphore(int(os.environ.get("LLM_CONCURRENCY", "4")))
    failed = 0
    CACHE.mkdir(exist_ok=True)
    progress = tqdm(total=len(todo), desc=task)
    with open(cache_path(task), "a", encoding="utf-8") as out:

        async def one(text: str, ex: list[tuple[str, str, str]]) -> None:
            nonlocal failed
            async with semaphore:
                try:
                    result = await ask(api, model, task, text, ex)
                except (APIError, ValidationError):
                    failed += 1
                else:
                    row = {"query": text, **result.model_dump()}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                progress.update()

        await asyncio.gather(*(one(t, ex) for t, ex in todo))
    progress.close()
    print(f"{task}: готово {len(todo) - failed}, не удалось {failed}, в кэше {len(answers(task))}")


def split_queries(split: str) -> tuple[list[str], pd.DataFrame]:
    if split == "valid":
        queries, _, rest = make_holdout()
    else:
        queries, _, rest = load_bench()
    return normalize_query(queries["search_query"]).tolist(), rest


async def check(task: str) -> None:
    api, model = client()
    queries, _, rest = make_holdout()
    text = "работа в клиненг"
    examples = few_shot([text], rest)[0]
    started = time.perf_counter()
    result = await ask(api, model, task, text, examples)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"{time.perf_counter() - started:.1f} с на запрос")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "run"])
    ap.add_argument("--task", choices=sorted(TASKS), default="kinds")
    ap.add_argument("--split", choices=["valid", "bench"], default="valid")
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые N запросов")
    args = ap.parse_args()
    if args.command == "check":
        asyncio.run(check(args.task))
        return
    texts, rest = split_queries(args.split)
    texts = texts[: args.limit]
    asyncio.run(run(args.task, texts, few_shot(texts, rest)))


if __name__ == "__main__":
    main()
