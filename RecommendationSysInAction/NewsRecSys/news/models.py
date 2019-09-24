from django.db import models

# Create your models here.


class Cate(models.Model):  # The table for categories of news.
    cate_id = models.BigIntegerField(verbose_name="ID", unique=True, blank=False)
    cate_name = models.CharField(blank=False, max_length=64, verbose_name="名字")

    def __str__(self):
        return self.cate_name
    objects = models.Manager()

    # define the name of the class in django backend
    class Meta:
        db_table = "cate"
        verbose_name_plural = "新闻类别表"


class News(models.Model):  # The table for original news
    news_id = models.BigIntegerField(verbose_name="ID", unique=True, blank=False, db_index=True)
    cate = models.ForeignKey(Cate, related_name="cate_news", on_delete=models.CASCADE)
    dt = models.DateTimeField(verbose_name="发表时间", blank=False)
    view_num = models.IntegerField(verbose_name="浏览次数", blank=True, default=0)
    comment_num = models.IntegerField(verbose_name="跟帖次数", blank=True, default=0)
    title = models.CharField(blank=False, max_length=100, verbose_name="标题")
    content = models.TextField(blank=False, verbose_name="正文")

    def __str__(self):
        return self.title

    objects = models.Manager()

    # define the name of the class in django backend
    class Meta:
        db_table = "news"
        verbose_name_plural = "新闻信息表"


class NewsHotness(models.Model):  # The table for hotness of news.
    news_id = models.BigIntegerField(verbose_name="ID", unique=True, blank=False, db_index=True)
    cate_id = models.ForeignKey(Cate, related_name="cate_news", on_delete=models.CASCADE)
    hotness = models.FloatField(verbose_name="热度", blank=False)

    def __str__(self):
        return self.news_id

    objects = models.Manager()

    class Meta:
        db_table = "news_hotness"
        verbose_name_plural = "新闻热度表"


class NewsTag(models.Model):  # The table for news's tag
    news_id = models.BigIntegerField(verbose_name="ID", unique=False, blank=False, db_index=True)
    tag = models.CharField(verbose_name="标签", max_length=64, unique=False, blank=False)

    def __str__(self):
        return f"{self.news_id}: {self.tag}"

    objects = models.Manager()

    class Meta:
        db_table = "news_tag"
        verbose_name_plural = "新闻标签表"


class NewsClick(models.Model):  # The table for the click for news
    news_id = models.BigIntegerField(verbose_name="ID", unique=False, blank=False)
    user = models.BigIntegerField(verbose_name="user_id", unique=False, blank=False)
    click_dt = models.DateTimeField(verbose_name="浏览时间", blank=False, unique=False)

    def __str__(self):
        return f"{self.user}: {self.news_id}"

    objects = models.Manager()

    class Meta:
        db_table = "news_click"
        verbose_name_plural = "新闻点击表"


class NewsSim(models.Model):  # The table for the similarity between each two news
    news_id_left = models.BigIntegerField(verbose_name="left_ID", unique=False, blank=False, db_index=True)
    news_id_right = models.BigIntegerField(verbose_name="right_ID", unique=False, blank=False)
    sim = models.FloatField(verbose_name="相似度", blank=True, default=0.)

    def __str__(self):
        return f"{self.news_id_left} v {self.news_id_right}: {self.sim}"

    objects = models.Manager()

    class Meta:
        db_table = "news_sim"
        verbose_name_plural = "新闻相似度表"

