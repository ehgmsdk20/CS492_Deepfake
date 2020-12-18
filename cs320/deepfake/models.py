from django.db import models
from django.conf import settings
from django import forms
from django.core.validators import FileExtensionValidator


# Create your models here.
def user_path(instance, filename): #파라미터 instance는 Photo 모델을 의미 filename은 업로드 된 파일의 파일 이름
    return instance.useremail+'/'+filename


class UploadFile(models.Model):
    useremail = models.EmailField(max_length=128, verbose_name="사용자 이메일", default="Enter")
    file = models.FileField(upload_to=user_path, verbose_name="동영상 파일", validators=[FileExtensionValidator(allowed_extensions=['mp4','mkv'])])
# Create your models here.
