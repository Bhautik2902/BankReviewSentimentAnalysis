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


class VisualiData:
    def __init__(self):
        self.bank_name = ""
        self.total_reviews = 0
        self.avg_rating = 0.0
        self.searched_st_service = ""
        self.searched_query = ""

        self.positive_reviews = []
        self.negative_reviews = []
        self.common_services = []

# class ServiceModel:
#     def __init__(self, name: str, positive: int, negative: int, neutral: int):
#         self.name = name
#         self.positive = positive
#         self.negative = negative
#         self.neutral = neutral
#
#     def to_dict(self):
#         return {
#             'name': self.name,
#             'positive': self.positive,
#             'negative': self.negative,
#             'neutral': self.neutral
#         }
#
#     @classmethod
#     def from_dict(cls, data):
#         return cls(data['name'], data['positive'], data['negative'], data['neutral'])
#
#     def __str__(self):
#         return f"Name: {self.name}"
