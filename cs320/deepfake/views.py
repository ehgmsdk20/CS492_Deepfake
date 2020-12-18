from django.shortcuts import render
from .forms import UploadFileForm
# Create your views here.
from django.views.generic.base import TemplateView
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import FormView

import os
import deepfake.MTCNN_video_face_detection_alignment as mtcnn
import deepfake.prep_binary_masks as prepro
import deepfake.FaceSwap_GAN_train_test as train
import deepfake.FaceSwap_GAN_video_conversion as video_convert
import smtplib
from email import encoders
from email.utils import formataddr
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


file_dir=os.path.dirname(os.path.abspath(__file__))

adminmail = #관리자 메일 주소
password = #관리자 비밀번호

smtp_gmail = smtplib.SMTP('smtp.gmail.com', 587)
smtp_gmail.ehlo()
smtp_gmail.starttls()
smtp_gmail.login(adminmail, password)


class IndexView(TemplateView): # TemplateView를 상속 받는다.
    template_name = 'deepfake/index.html'

class UploadView(FormView): # TemplateView를 상속 받는다.
    form_class = UploadFileForm
    template_name = 'deepfake/uploadpage.html'

    @method_decorator(csrf_exempt)
    def upload_file(request):
        if(request.method == 'POST'):
            form = UploadFileForm(request.POST, request.FILES)
     
            if form.is_valid():
                email_addr = form.cleaned_data['useremail']
                uploaded_file = form.save(commit=False)
                uploaded_file.ip = request.META['REMOTE_ADDR']
                uploaded_file.save()
                mtcnn.all_thing(form)
                prepro.all_thing(form)
                train.all_thing(form)
                video_convert.all_thing(form)
                output_name = os.path.join("/home/ubuntu/cs320/media", email_addr, "OUTPUT_VIDEO.mp4")
                try:
                    message = MIMEMultipart("mixed")

                    # 메일 송/수신 옵션 설정
                    message.set_charset('utf-8')
                    message['From'] = adminmail
                    message['To'] = email_addr
                    message['Subject'] = '[CS492] Deepfake Result'
                    # 메일 콘텐츠 - 내용
                    body = '''
                    <h2>please check attatchment.</h1>
                    '''
                    bodyPart = MIMEText(body, 'html', 'utf-8')
                    message.attach( bodyPart )
                    # 메일 콘텐츠 - 첨부파일
                    attachments = [
                    output_name
                    ]
    
                    for attachment in attachments:
                        attach_binary = MIMEBase("application", "octect-stream")
                        try:
                            binary = open(attachment, "rb").read() # read file to bytes
    
                            attach_binary.set_payload( binary )
                            encoders.encode_base64( attach_binary ) # Content-Transfer-Encoding: base64

                            filename = os.path.basename( attachment )
                            attach_binary.add_header("Content-Disposition", 'attachment', filename=('utf-8', '', filename))
                            message.attach( attach_binary )
                        except Exception as e:
                            print(e)
                    # 메일 발송
                    smtp_gmail.sendmail(adminmail, email_addr, message.as_string())
                    smtp_gmail.quit()


                except:
                    return HttpResponse('''
                            <h1> 메일 전송 오류 </h1>
                            ''')

                return HttpResponse('''
                            <h1> 업로드 완료 </h1>
                            ''')
                
            else:
                return HttpResponse('''
                            <h1> 입력이 올바르지 않습니다. </h1>
                            ''')
        else:
            return render(request, 'new.html')
# Create your views here.
