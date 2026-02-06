from sqlalchemy import Column, Integer, String, DateTime, Text

class Feedback():
    
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    img_id = Column(Integer)
    feedback_date = Column(DateTime)
    predicted_class = Column(String)
    diagnostic = Column(String)
    comment = Column(Text)

    def __init__(
        self,
        img_id,
        feedback_date,
        predicted_class,
        diagnostic,
        comment,
    ):
        self.img_id = img_id
        self.feedback_date = feedback_date
        self.predicted_class = predicted_class
        self.diagnostic = diagnostic
        self.comment = comment

    def get_img_id(self):
        return self.img_id

    def set_img_id(self, value):
        self.img_id = value

    def get_feedback_date(self):
        return self.feedback_date

    def set_feedback_date(self, value):
        self.feedback_date = value

    def get_predicted_class(self):
        return self.predicted_class

    def set_predicted_class(self, value):
        self.predicted_class = value

    def get_diagnostic(self):
        return self.diagnostic

    def set_diagnostic(self, value):
        self.diagnostic = value

    def get_comment(self):
        return self.comment

    def set_comment(self, value):
        self.comment = value
