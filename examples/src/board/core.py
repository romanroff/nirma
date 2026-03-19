from langchain.tools import tool
from typing import Callable
from pydantic import Field
from ..model import Model
from .models import *
from .utils import *

TOOLS = {
    'add_tasks',
    'get_task',
    'get_tasks',
    'update_status',
    'update_priority',
    'add_result',
    'get_results'
}

class Board(Model):
    iteration : int = Field(default=0)
    tasks : list[Task] = Field(default=[])
    results : list[Result] = Field(default=[])
    
    @property
    def tools(self) -> list[Callable]:
        return [tool(getattr(self, name)) for name in TOOLS]

    def update_iteration(self) -> int:
        iteration = self.iteration + 1
        self.iteration = iteration
        return iteration
    
    #########
    # TASKS #
    #########

    def add_tasks(self, tasks : list[BaseTask]) -> list[str]:
        """
        Добавляет перечень задач на доску задач
        Возвращает список ID задач в порядке добавления.
        """
        with lock:
            tasks = [Task(
                id=get_id(),
                created_at=self.iteration,
                **task.model_dump()
            ) for task in tasks]
            self.tasks = tasks
            return [t.id for t in self.tasks]
    
    def get_task(self, task_id : str) -> Task | None:
        """
        Возвращает задачу по ID.
        Возвращает None, если задача не найдена.
        """
        tasks = list(filter(lambda t : t.id == task_id, self.tasks))
        if len(tasks) == 0:
            return None
        return tasks[0]
    
    def get_tasks(self, status : Status | None = None, priority : Priority = Priority.TRIVIAL) -> list[Task]:
        """
        Возвращает перечень доступных задач с фильтром:
        - status : Status | None - только заданный статус (если None, фильтр не применяется). По умолчанию None.
        - priority : Priority - заданный приоритет и выше (по умолчанию Priority.TRIVIAL) 
        Если фильтр не указан, вернется весь перечень задач.
        """
        tasks = self.tasks
        if status is not None:
            tasks = [t.status == status for t in tasks]
        return [t for t in tasks if t.priority >= priority]
    
    def update_status(self, task_id : str, status : Status) -> Task | None:
        """
        Обновляет текущий статус задачи.
        Возвращает None, если задача с таким ID не найдена.
        Возвращает текущее состояние задачи.
        ВНИМАНИЕ: сверяйте желаемое и текущее состояние задачи.
        """
        task = self.get_task(task_id)
        if task is None:
            return None
        status_before = task.status
        with lock:
            status_after = task.status
            if status_before == status_after:
                task.status = status
            return task

    def update_priority(self, task_id : str, priority : Priority) -> Task | None:             
        """
        Обновляет текущий приоритет задачи.
        Возвращает None, если задача с таким ID не найдена.
        Возвращает текущее состояние задачи.
        ВНИМАНИЕ: сверяйте желаемое и текущее состояние задачи.
        """
        task = self.get_task(task_id)
        if task is None:
            return None
        priority_before = task.priority
        with lock:
            priority_after = task.priority
            if priority_before == priority_after:
                task.priority = priority
            return task
        
    ###########
    # RESULTS #
    ###########
    
    def add_result(self, task_id : str, result : BaseResult) -> Result | None:
        """
        Добавляет результат к задаче. 
        Возвращает None, если задача с таким ID не найдена.
        """
        task = self.get_task(task_id)
        if task is None:
            return None
        
        with lock:
            result = Result(
                task_id=task.id,
                created_at=self.iteration,
                **result.model_dump()
            )
            self.results.append(result)
            return result
        
    def get_results(self, task_id : str) -> list[Result] | None:
        """
        Возвращает результаты, относящиеся к задаче. 
        Возвращает None, если задача с таким ID не найдена.
        """               
        task = self.get_task(task_id)
        if task is None:
            return None
        results = [c for c in self.results if c.task_id == task.id]
        return results