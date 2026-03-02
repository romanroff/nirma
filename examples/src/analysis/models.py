from pydantic import Field
from ..models import Model, QueryModel

class SocioEconomicModel(Model):
    """
    Социально-экономический анализ
    """
    demography : str = Field(description='Демография')
    productivity : str = Field(description='Производительность труда')
    grdp : str = Field(description='Структура и динамика валового городского продукта')
    labor : str = Field(description='Рынок труда')
    real_estate : str = Field(description='Рынок жилья и коммерческой недвижимости')
    budget : str = Field(description='Бюджетная обеспеченность')

class SpatialModel(Model):
    """
    Пространственный анализ
    """
    functional_organization : str = Field(description='Функциональная организация территории')
    land_tenure : str = Field(description='Структура землевладения')
    ecology : str = Field(description='Природно-экологический каркас')
    architecture : str = Field(description='Архитектурно-градостроительная среда')

class TransportModel(Model):
    """
    Транспортный анализ
    """
    transport : str = Field(description='Городской и внешний пассажирский и грузовой транспорт')
    parking : str = Field(description='Парковки')
    pedestrian : str = Field(description='Пешеходные зоны')

class EngineeringModel(Model):
    """
    Анализ инженерной инфраструктуры
    """
    water : str = Field(description='Водоснабжение и водоотведение')
    energy : str = Field(description='Энергоснабжение')
    heat : str = Field(description='Теплоснабжение')

class AnalysisModel(Model): #XVII. КОНЦЕПЦИЯ ПРОСТРАНСТВЕННОГО РАЗВИТИЯ МО (МАСТЕР-ПЛАН)
    """
    Диагностика экономического и пространственного состояния территории
    """
    socio_economic : SocioEconomicModel | None = Field(default=None, description='Социально-экономический анализ')
    spatial : SpatialModel | None = Field(default=None, description='Пространственный анализ')
    transport : TransportModel | None = Field(default=None, description='Транспортный анализ')
    engineering : EngineeringModel | None = Field(default=None, description='Анализ инженерной инфраструктуры')

class AnalysisState(AnalysisModel):
    query : QueryModel