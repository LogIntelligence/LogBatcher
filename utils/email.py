import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

class Email_send:
    def __init__(self, Theme):
        self.Theme = Theme
        self.sender_email = "1802123579@qq.com"
        self.password = "vifeqvrfaomredhe"
        self.receiver_email = "1802123579@qq.com"

    def send_table(self,table):

        # 创建一个MIMEMultipart对象，用于包含邮件内容
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = f"Result for {self.Theme}"  # 设置邮件主题

        # 添加邮件正文
        body = "experiment output table as follows:"
        message.attach(MIMEText(body + "\n\n" + table, "html"))

        # 连接到SMTP服务器
        with smtplib.SMTP("smtp.qq.com", 587) as server:
            server.starttls()  # 启用TLS加密
            server.login(self.sender_email, self.password)  # 登录到发件人邮箱

            # 发送邮件
            server.send_message(message)
        print("Email sent successfully!")
