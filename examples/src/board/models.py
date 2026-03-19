from ..model import Model
from pydantic import Field
from enum import Enum, IntEnum

class Status(Enum):
    TODO='todo'
    """Ожидает исполнителей"""
    IN_PROGRESS='in_progress'
    """Выполняется"""
    ON_PAUSE='on_pause'
    """На паузе"""
    REVIEW='review'
    """Ожидается проверка результатов"""
    DONE='done' 
    """Результаты приняты"""
    CANCELLED='cancelled'
    """Отменена (потеряла актуальность, невозможно выполнить, разбита на подзадачи)"""

class Priority(IntEnum):
    TRIVIAL=1
    """Необязательная задача"""
    LOW=2
    """Низкий приоритет"""
    MEDIUM=3
    """Средний приоритет"""
    HIGH=4
    """Высокий приоритет"""
    CRITICAL=5
    """Критически важная задача"""

class BaseTask(Model):
    name : str = Field(description='Название задачи')
    description : str = Field(description='Описание задачи')
    priority : Priority = Field(description='Приоритет задачи')

class Task(BaseTask):
    id : str = Field(description='ID задачи. Создается автоматически')
    status : Status = Field(default=Status.TODO, description='Статус задачи. Создается автоматически')
    created_at : int = Field(description='Номер итерации, на которой была создана задача. Создается автоматически')
    
class BaseResult(Model):
    author : str = Field(description='Автор результата')
    content : str = Field(description="Содержание результата")

class Result(BaseResult):
    id : str = Field(description='ID результата. Создается автоматически')
    task_id : str = Field(description='ID задачи. Создается автоматически')
    created_at : int = Field(description='Номер итерации, на которой был создан комментарий. Создается автоматически')

__all__=[
    'Status',
    'Priority',
    'BaseTask',
    'Task',
    'BaseResult',
    'Result',
]