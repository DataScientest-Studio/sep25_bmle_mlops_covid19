from sqlalchemy import Column, Integer, String, DateTime


class ImageDataset():
    
    __tablename__ = "images_dataset"

    id = Column(Integer, primary_key=True)
    image_url = Column(String)
    mask_url = Column(String)
    class_type = Column(String)
    injection_date = Column(DateTime)
    created_at = Column(DateTime)

    def __init__(
        self,
        image_url,
        mask_url,
        class_type,
        injection_date,
        created_at,
    ):
        self.image_url = image_url
        self.mask_url = mask_url
        self.class_type = class_type
        self.injection_date = injection_date
        self.created_at = created_at

    def get_image_url(self):
        return self.image_url

    def set_image_url(self, value):
        self.image_url = value

    def get_mask_url(self):
        return self.mask_url

    def set_mask_url(self, value):
        self.mask_url = value

    def get_class_type(self):
        return self.class_type

    def set_class_type(self, value):
        self.class_type = value

    def get_injection_date(self):
        return self.injection_date

    def set_injection_date(self, value):
        self.injection_date = value

    def get_created_at(self):
        return self.created_at

    def set_created_at(self, value):
        self.created_at = value
