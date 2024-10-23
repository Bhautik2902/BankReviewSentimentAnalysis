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


class VisuliData(models.Model):
    bank_name = models.CharField(max_length=20)
    total_reviews = models.IntegerField()
    avg_rating = models.FloatField()
    searched_st_service = models.CharField(max_length=50)
    searched_query = models.TextField()

    common_services = models.JSONField(null=True, blank=True)
    positive_reviews = models.TextField(null=True, blank=True)
    negative_reviews = models.TextField(null=True, blank=True)

    def add_instance(self, instance):
        if not self.common_services:
            self.instances_list = []
        self.instances_list.append(instance.to_dict())
        self.save()

    def get_instances(self):
        return [ServiceModel.from_dict(item) for item in self.instances_list]

    def get_review_list(self, listtype: int):
        if self.positive_reviews and listtype == 1:
            return self.positive_reviews.split(',')
        elif self.negative_reviews and listtype == 0:
            return self.negative_reviews.split(',')
        return []

    def set_positive_services_list(self, positive_reviews):
        self.services = ','.join(positive_reviews)

    def set_negative_services_list(self, negative_reviews):
        self.services = ','.join(negative_reviews)


class ServiceModel:
    def __init__(self, name: str, positive: int, negative: int, neutral: int):
        self.name = name
        self.positive = positive
        self.negative = negative
        self.neutral = neutral

    def to_dict(self):
        return {
            'name': self.name,
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['name'], data['positive'], data['negative'], data['neutral'])

    def __str__(self):
        return f"Name: {self.name}"
