# Car-Detection using Raspberry Pi
 a prototype through raspberry pi shows whether the captured video consists of a car or not and counts them.

# Prerequisites

## Hardware:


A programmable computer hardware (Raspberry Pi)

Camera Module (Raspberry Pi Camera Module)


## Software:


any kernel can configuration the hardware (Raspberry Pi OS)

### Libraries:


Python 3.9.4 (or any Default/Update installed Python on Pi OS)

scipy (SciPy: Scientific Library for Python)

numpy (NumPy is the fundamental package for array computing with Python.)

opencv-python


# Installing

Enter the commands to install all the libraries required for this project.


    sudo apt-get install python-pip python-dev build-essential

    sudo pip install --upgrade pip

    sudo pip install -r requirements.txt


After Installing all libraries connect the camera (picamera) to raspberry pi.

Execution
Once all the components are integrated with pi then run the following command for execution.

    python main.py

Expected Output

پس از اجرای موفق برنامه در صورتی که ماشین از جلوی دوربین عبور کند باید تشخیص داده شود و در فایل 'software.log' نوشته شود.


# Software Components


function to get the centroid of the object

function to detect vehicle/moving object 

function to normalize the image so that the entire object has the same rgb value

initialize color class

convert the rgb color to lab colors (update the L*a*b* array and the color names list)

function to label the car’s lab color to a particular color class

initialize the minimum distance found thus far

loop over the known L*a*b* color values

compute the distance between the current L*a*b* color value and the mean of the image	
      
if the distance is smaller than the current distance, then update the minimum distance

return the name of the color with the smallest distance

initialize background object used for background elimination

initialize frame counter to run the program in a while loop:

resize the frame

creating a copy of the frame

applying background elimination

additional image processing

fill parts of image with small area

remove noise

dilate to merge adjacent objects in the feature space

area threshold to further remove noise 

initialize output color list

detect contour and calculate the coordinates of the contours

loop over each detected contour 

extract the regions that contains car features

normalize the region so as to obtain a uniform color.

reshape the image to a linear form with 3-channels

initialize centroids

apply k-means clustering to detect prominent color in the object

detect the centroid with densest cluster

get the label of the car color

draw rectangle over the detected car

label each rectangle with the detected color of the car

open file to write the output of each frame

write into the file every 10 frames

write detected car number 




# Methodologies


پس از تشخیص و حذف پس زمینه، عکس را resize  میکنیم، یک کپی از frame تهیه میکنیم تا پردازش ها روی آن انجام شود. روی کپی تصویر پس زمینه را حذف میکنیم. قسمت هایی از عکس که مساحت کمی دارند و به خاطر نویز خالی مانده اند را با 1 پر میکنیم. در تصویر بدست آمده حذف نویز انجام میدهیم. یک فیلتر dilation روی عکس اعمال میکنیم تا نواحی مجاور و نزدیک که به خاطر نویز در تصویر از هم جدا هستند با هم ادغام شوند. پس از ادغام نواحی نزدیک بهم نواحی درون feature map که مساحت کمی دارند را حذف میکنیم. لیستی از رنگ های ممکن برای خروجی تهیه میکنیم. تمام contour های درون feature map را استخراج میکنیم و مرکز آنها را بدست می آوریم.  نواحی درون هر contour که شامل ویژگی های رنگی از ماشین است را استخراج می کنیم و آن را normalize می کنیم تا رنگ یکسانی در هر ناحیه حاصل شود. شکل ماتریسی عکس را تغییر میدهیم تا 3 کانال رنگی RGB داشته باشد. روی کانتور های بدست آمده از feature map ، از الگوریتم k means برای بدست آوردن رنگ غالب استفاده میکنیم و از cluster با بالاترین density استفاده میکنیم که رنگ ماشین تخمین زده شود. با توجه به رنگ بدست آمده از کانتور ها رنگ ماشین را از لیست رنگ های شناخته شده انتخاب میکنیم. دور ماشین یک bounding box میگذاریم و label آن را رنگ شناسایی شده برای ماشین قرار می دهیم. هر یک ساعت تعداد ماشین های عبور کرده را در فایل log ذخیره میکنیم.


# Development

this Software by Designed in a Functional Programming


…


## تنظیم لاگ برنامه


    logging.basicConfig(filename='software.log',
                            format='[%(funcName)s] - %(levelname)s [%(asctime)s] %(message)s', level=logging.DEBUG)


در خط 12-13
در صورت نیاز به خروج از مرحله دیباک
لول را از دیباگ به اینفو تغییر دهید

    level=logging.info

در خط 31

    min_contour_width=35, min_contour_height=30

برای تعین طول و عرض حد فاصل مرئی اشیا برای تشخیص است که وابسته به محل قرارگیری دوربین می باشد

با استفاده از دستور زیر در خط 137-140

    cap = cv2.VideoCapture(0)

    #cap = cv2.VideoCapture('traffic_video.mp4')


میتوانید ورودی دستگاه را تغییر دهید به این صورت که مقدار
0 و -1 برای هر یک از دوربین های متصل به دستگاه می باشد که در حال حاضر مقدار 0 را در رزبری پای دارد یا آدرس هر نوع ویدئویی را به عنوان ورودی به برنامه دارید

در خط 152 میتوانید با استفاده از دستور
		frame=cv2.resize(frame,(720,1280))
و تغییر 720 و 1280 اندازه فریم تصویر را وابسته به کیفیت دوربین و دقتی که نیاز هست تنظیم کنید
توجه کنید هرچه برنامه عرض و ارتفاع کمتری به عنوان تصویر ورودی داشته باشد سریع تر است 
که وابسته به موقعیت قرار گیری تغییر میکند

مهم ترین قسمت تغییر کد برای شمارش و دقت شمارش 

خط 165 با دستور

    bg= cv2.erode(bg,kernel,iterations = 30) #8  => best 30 for sample 1 and 2 -------- 15 for sample 3

میباشد

که برای مثال بهترین مقدار 30 بود


…




# Acknowledgments


python developers and python community

python

Opencv-python


@Amir-Mehrpanah


# References

Python 3.9.4 documentation - https://docs.python.org/3/

Open Source Computer Vision documentation - https://docs.opencv.org/master/


...

