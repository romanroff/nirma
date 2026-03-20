import os
import sys

from src import (
    AgentTask,
    Store,
    create_document_research_agent,
    create_web_research_agent,
)


def print_result(title: str, result) -> None:
    print(f"\n=== {title} ===")
    print(result.model_dump_json(indent=2, ensure_ascii=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    web_agent = create_web_research_agent()
    web_task = AgentTask(
        kind="web_research",
        query=(
            "Как можно развивать Гатчину с учетом ее роли в агломерации Санкт-Петербурга? "
            "Кратко назови 5 стратегических направлений и отдельно отметь транспорт, "
            "экономику и городскую среду."
        ),
    )
    print_result("Веб-исследование", web_agent.execute(web_task))

    document_path = os.getenv("NIRMA_DOCUMENT_PATH")
    if not document_path:
        print(
            "\nУкажи NIRMA_DOCUMENT_PATH с путем к PDF или DOCX, чтобы "
            "запустить пример document-агента."
        )
        return

    store = Store(document_path)
    document_agent = create_document_research_agent(store=store)
    document_task = AgentTask(
        kind="document_research",
        query=(
            "Какие ключевые стратегические приоритеты и направления развития "
            "упоминаются в этом документе? Кратко перечисли основные пункты."
        ),
    )
    print_result("Анализ документа", document_agent.execute(document_task))


if __name__ == "__main__":
    main()
