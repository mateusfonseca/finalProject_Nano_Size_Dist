import base64
from tempfile import TemporaryFile, NamedTemporaryFile

import cv2
import numpy as np
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import generic
from django.views.decorators.csrf import csrf_protect
from django.views.generic import TemplateView


class IndexView(TemplateView):
    template_name = "nanoer/index.html"

    def post(self, request):
        # print(vars(request.FILES['img'].file))
        # print(request.FILES['img'].read())
        # image = "data:image/*;base64," + base64.b64encode(request.FILES['img'].read()).decode()

        path = request.FILES['img'].temporary_file_path()
        path_converted = path+'_converted.jpg'

        image = cv2.imread(path)

        image = cv2.resize(image, (500, 500))
        image = cv2.medianBlur(image, 5)
        cimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image=cimg, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30,
                                  minRadius=5, maxRadius=100)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        # cimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(path_converted, cimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        img = cv2.imread(path_converted)
        _, im_arr = cv2.imencode('.jpg', img)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        image = "data:image/*;base64," + im_b64.decode()

        print("Number of circles: " + str(len(circles[0])))

        import pandas as pd
        radius_list = lambda circle: circle[0:, 2]
        df = pd.DataFrame(radius_list(circles[0]))
        ax = df.plot.hist(edgecolor='black', range=[5, 50], bins=30).get_figure()

        with NamedTemporaryFile() as f:
            ax.savefig(f)
            chart = cv2.imread(f.name)
            _, chart_arr = cv2.imencode('.jpg', chart)
            chart_bytes = chart_arr.tobytes()
            chart_b64 = base64.b64encode(chart_bytes)
            chart = "data:image/*;base64," + chart_b64.decode()

        print(f'Image: {image}')
        print(f'Chart: {chart}')

        # print(request.FILES['img'].temporary_file_path())
        # print(base64.b64encode(cv2.imread(request.FILES['img'].temporary_file_path())).decode())
        # image = request.FILES['img'].temporary_file_path()
        # print(image)

        context = {'image': image, 'chart': chart}

        return render(request, self.template_name, context)
