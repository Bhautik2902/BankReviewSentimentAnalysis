from django.db import models


class Review(models.Model):
    # Non-nullable fields
    index = models.IntegerField()
    source = models.CharField(max_length=50)
    bank = models.CharField(max_length=100)
    title = models.TextField()
    review_text = models.TextField()
    rating = models.FloatField(null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    url = models.URLField(max_length=500, null=True, blank=True)
    source_type = models.CharField(max_length=50, null=True, blank=True)
    review_sentiment = models.CharField(max_length=50)
    sentiment_score = models.IntegerField()

    def __str__(self):
        return f"{self.bank} - {self.source} - {self.review_sentiment}"


class ServiceModel:
    def __init__(self):
        self.name = ""
        self.pos_count = 0
        self.neg_count = 0
        self.neu_count = 0

    def to_dict(self):
        return {
            "name": self.name,
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "neu_count": self.neu_count
        }


class VisualiData:
    def __init__(self):
        self.bank_name = ""
        self.total_reviews = 0
        self.avg_rating = 0.0
        self.searched_st_service = ""
        self.searched_query = ""

        self.positive_reviews = []
        self.negative_reviews = []
        self.common_services = []  # list of ServiceModel instance

        self.pos_count = 0
        self.neg_count = 0
        self.neu_count = 0
        self.service_at_other_banks = {}
        self.curr_bank_list = []

        self.positive_word_list = []
        self.negative_word_list = []

    def to_dict(self):
        return {
            "bank_name": self.bank_name,
            "total_reviews": self.total_reviews,
            "avg_rating": self.avg_rating,
            "searched_st_service": self.searched_st_service,
            "searched_query": self.searched_query,
            "positive_reviews": self.positive_reviews,
            "negative_reviews": self.negative_reviews,
            "common_services": [service.to_dict() for service in self.common_services],  # Serialize ServiceModel
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "neu_count": self.neu_count,
            "service_at_other_banks": self.service_at_other_banks,
            "curr_bank_list": self.curr_bank_list,
            "positive_word_list": self.positive_word_list,
            "negative_word_list": self.negative_word_list
        }
