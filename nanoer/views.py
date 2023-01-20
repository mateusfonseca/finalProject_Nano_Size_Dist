import base64

import cv2
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import generic
from django.views.generic import TemplateView


class IndexView(TemplateView):
    template_name = "nanoer/index.html"

    def post(self, request):
        # print(vars(request.FILES['img'].file))
        # print(request.FILES['img'].read())
        # image = "data:image/*;base64," + base64.b64encode(request.FILES['img'].read()).decode()

        path = request.FILES['img'].temporary_file_path()
        path_converted = path+'_converted.jpg'
        image_tiff = cv2.imread(path)
        cv2.imwrite(path_converted, image_tiff, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        img = cv2.imread(path_converted)
        _, im_arr = cv2.imencode('.jpg', img)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        image = "data:image/*;base64," + im_b64.decode()

        # print(request.FILES['img'].temporary_file_path())
        # print(base64.b64encode(cv2.imread(request.FILES['img'].temporary_file_path())).decode())
        # image = request.FILES['img'].temporary_file_path()
        # print(image)

        return render(request, self.template_name, {'image': image})
