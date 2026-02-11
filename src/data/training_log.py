from sqlalchemy import Column, Integer, Float, DateTime

class TrainingLog():
    
    __tablename__ = "training_log"

    last_training = Column(DateTime, primary_key=True)
    training_size = Column(Integer)
    validation_size = Column(Integer)
    epochs_number = Column(Integer)
    accuracy = Column(Float)
    class_0_precision = Column(Float)
    class_0_recall = Column(Float)
    class_0_f1 = Column(Float)
    class_1_precision = Column(Float)
    class_1_recall = Column(Float)
    class_1_f1 = Column(Float)
    true_class_0 = Column(Integer)
    false_class_0 = Column(Integer)
    true_class_1 = Column(Integer)
    false_class_1 = Column(Integer)

    def __init__(
        self,
        last_training,
        training_size,
        validation_size,
        epochs_number,
        accuracy,
        class_0_precision,
        class_0_recall,
        class_0_f1,
        class_1_precision,
        class_1_recall,
        class_1_f1,
        true_class_0,
        false_class_0,
        true_class_1,
        false_class_1,
    ):
        self.last_training = last_training
        self.training_size = training_size
        self.validation_size = validation_size
        self.epochs_number = epochs_number
        self.accuracy = accuracy
        self.class_0_precision = class_0_precision
        self.class_0_recall = class_0_recall
        self.class_0_f1 = class_0_f1
        self.class_1_precision = class_1_precision
        self.class_1_recall = class_1_recall
        self.class_1_f1 = class_1_f1
        self.true_class_0 = true_class_0
        self.false_class_0 = false_class_0
        self.true_class_1 = true_class_1
        self.false_class_1 = false_class_1

    def get_last_training(self):
        return self.last_training

    def set_last_training(self, value):
        self.last_training = value

    def get_training_size(self):
        return self.training_size

    def set_training_size(self, value):
        self.training_size = value

    def get_validation_size(self):
        return self.validation_size

    def set_validation_size(self, value):
        self.validation_size = value

    def get_epochs_number(self):
        return self.epochs_number

    def set_epochs_number(self, value):
        self.epochs_number = value

    def get_accuracy(self):
        return self.accuracy

    def set_accuracy(self, value):
        self.accuracy = value

    def get_class_0_precision(self):
        return self.class_0_precision

    def set_class_0_precision(self, value):
        self.class_0_precision = value

    def get_class_0_recall(self):
        return self.class_0_recall

    def set_class_0_recall(self, value):
        self.class_0_recall = value

    def get_class_0_f1(self):
        return self.class_0_f1

    def set_class_0_f1(self, value):
        self.class_0_f1 = value

    def get_class_1_precision(self):
        return self.class_1_precision

    def set_class_1_precision(self, value):
        self.class_1_precision = value

    def get_class_1_recall(self):
        return self.class_1_recall

    def set_class_1_recall(self, value):
        self.class_1_recall = value

    def get_class_1_f1(self):
        return self.class_1_f1

    def set_class_1_f1(self, value):
        self.class_1_f1 = value

    def get_true_class_0(self):
        return self.true_class_0

    def set_true_class_0(self, value):
        self.true_class_0 = value

    def get_false_class_0(self):
        return self.false_class_0

    def set_false_class_0(self, value):
        self.false_class_0 = value

    def get_true_class_1(self):
        return self.true_class_1

    def set_true_class_1(self, value):
        self.true_class_1 = value

    def get_false_class_1(self):
        return self.false_class_1

    def set_false_class_1(self, value):
        self.false_class_1 = value
