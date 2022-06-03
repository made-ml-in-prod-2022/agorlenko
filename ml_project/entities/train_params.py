from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    options: dict
    model_type: str = field(default='LogisticRegression')
