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

