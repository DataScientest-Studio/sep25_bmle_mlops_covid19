from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Parameters(Base):
    __tablename__ = "parameters"

    validity_date = Column(DateTime, primary_key=True)
    retraining_trigger_ratio = Column(Float)
    img_width = Column(Integer)
    img_height = Column(Integer)
    gray_mode = Column(Boolean)
    batch_size = Column(Integer)
    train_size = Column(Float)
    random_state = Column(Integer)
    oversampling_factor = Column(Integer)
    optimizer_name = Column(String)
    loss_cat = Column(String)
    metrics = Column(String)
    es_patience = Column(Integer)
    es_min_delta = Column(Float)
    es_mode = Column(String)
    es_monitor = Column(String)
    rlrop_patience = Column(String)
    rlrop_monitor = Column(String)
    rlrop_min_delta = Column(Float)
    rlrop_factor = Column(Float)
    rlrop_cooldown = Column(Integer)
    nb_layer_to_freeze = Column(Integer)

    def __init__(
        self,
        validity_date,
        retraining_trigger_ratio,
        img_width,
        img_height,
        gray_mode,
        batch_size,
        train_size,
        random_state,
        oversampling_factor,
        optimizer_name,
        loss_cat,
        metrics,
        es_patience,
        es_min_delta,
        es_mode,
        es_monitor,
        rlrop_patience,
        rlrop_monitor,
        rlrop_min_delta,
        rlrop_factor,
        rlrop_cooldown,
        nb_layer_to_freeze,
    ):
        self.validity_date = validity_date
        self.retraining_trigger_ratio = retraining_trigger_ratio
        self.img_width = img_width
        self.img_height = img_height
        self.gray_mode = gray_mode
        self.batch_size = batch_size
        self.train_size = train_size
        self.random_state = random_state
        self.oversampling_factor = oversampling_factor
        self.optimizer_name = optimizer_name
        self.loss_cat = loss_cat
        self.metrics = metrics
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.es_mode = es_mode
        self.es_monitor = es_monitor
        self.rlrop_patience = rlrop_patience
        self.rlrop_monitor = rlrop_monitor
        self.rlrop_min_delta = rlrop_min_delta
        self.rlrop_factor = rlrop_factor
        self.rlrop_cooldown = rlrop_cooldown
        self.nb_layer_to_freeze = nb_layer_to_freeze

    # ---- getters / setters ----

    def get_validity_date(self):
        return self.validity_date

    def set_validity_date(self, value):
        self.validity_date = value

    def get_retraining_trigger_ratio(self):
        return self.retraining_trigger_ratio

    def set_retraining_trigger_ratio(self, value):
        self.retraining_trigger_ratio = value

    def get_img_width(self):
        return self.img_width

    def set_img_width(self, value):
        self.img_width = value

    def get_img_height(self):
        return self.img_height

    def set_img_height(self, value):
        self.img_height = value

    def get_gray_mode(self):
        return self.gray_mode

    def set_gray_mode(self, value):
        self.gray_mode = value

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, value):
        self.batch_size = value

    def get_train_size(self):
        return self.train_size

    def set_train_size(self, value):
        self.train_size = value

    def get_random_state(self):
        return self.random_state

    def set_random_state(self, value):
        self.random_state = value

    def get_oversampling_factor(self):
        return self.oversampling_factor

    def set_oversampling_factor(self, value):
        self.oversampling_factor = value

    def get_optimizer_name(self):
        return self.optimizer_name

    def set_optimizer_name(self, value):
        self.optimizer_name = value

    def get_loss_cat(self):
        return self.loss_cat

    def set_loss_cat(self, value):
        self.loss_cat = value

    def get_metrics(self):
        return self.metrics

    def set_metrics(self, value):
        self.metrics = value

    def get_es_patience(self):
        return self.es_patience

    def set_es_patience(self, value):
        self.es_patience = value

    def get_es_min_delta(self):
        return self.es_min_delta

    def set_es_min_delta(self, value):
        self.es_min_delta = value

    def get_es_mode(self):
        return self.es_mode

    def set_es_mode(self, value):
        self.es_mode = value

    def get_es_monitor(self):
        return self.es_monitor

    def set_es_monitor(self, value):
        self.es_monitor = value

    def get_rlrop_patience(self):
        return self.rlrop_patience

    def set_rlrop_patience(self, value):
        self.rlrop_patience = value

    def get_rlrop_monitor(self):
        return self.rlrop_monitor

    def set_rlrop_monitor(self, value):
        self.rlrop_monitor = value

    def get_rlrop_min_delta(self):
        return self.rlrop_min_delta

    def set_rlrop_min_delta(self, value):
        self.rlrop_min_delta = value

    def get_rlrop_factor(self):
        return self.rlrop_factor

    def set_rlrop_factor(self, value):
        self.rlrop_factor = value

    def get_rlrop_cooldown(self):
        return self.rlrop_cooldown

    def set_rlrop_cooldown(self, value):
        self.rlrop_cooldown = value

    def get_nb_layer_to_freeze(self):
        return self.nb_layer_to_freeze

    def set_nb_layer_to_freeze(self, value):
        self.nb_layer_to_freeze = value
