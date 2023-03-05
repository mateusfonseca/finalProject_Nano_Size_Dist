import os
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import generic
from django.views.generic import TemplateView
from matplotlib import pyplot as plt
from pytesseract import pytesseract

from finalProject.settings import MEDIA_ROOT, MEDIA_URL
from nanoer.models import Project, File, Analysis


# This file defines all the views in the internal app nanoer.
# The views are classes and functions that respond to web requests with appropriate web responses.
# They invoke the templates that will be rendered in return (if applicable) and handle any errors
# that may arise during the handling of the requests.


# crop uploaded image so that the actual image and the legend
# can be handled separately
def crop_image(img):
    max_y = img.shape[0] - 1  # highest pixel position on the y-axis
    max_x = img.shape[1] - 1  # highest pixel position on the x-axis

    div = max_y

    # from the bottom-right corner up on the legend
    # until it is no longer white, where the legend ends
    for p in range(max_y, 0, -1):
        if np.array_equal(img[p][max_x], [255, 255, 255]):
            div = p
        else:
            break  # div holds the last white pixel position on the y-axis

    image = img[:div, ]  # only the image
    legend = img[div:, int(max_x / 2):]  # only the right-half portion of the legend

    return image, legend


# calculate the factor that each pixel needs to be scaled by
# in order to get the actual size of the nanoparticles (in nanometers)
def calculate_scale(img):  # this image is just the right-half portion of the legend
    max_y = img.shape[0] - 1  # highest pixel position on the y-axis
    max_x = img.shape[1] - 1  # highest pixel position on the x-axis
    x_ref = int(max_x * 1 / 4)  # x-axis reference placed on top of the scale bar

    bar_y = 0

    # from the top of the image, downwards, aligned with x_ref
    # until it is no longer white, where the scale bar starts
    for p in range(max_y):
        if np.array_equal(img[p][x_ref], [255, 255, 255]):
            continue
        else:
            bar_y = p  # y-axis position of the scale bar
            break

    bar_x_left = 0
    bar_x_right = max_x

    # from x_ref, to the left on the x
    # until it becomes white, where the scale bar ends
    for p in range(x_ref, 0, -1):
        if np.any(np.not_equal(img[bar_y][p], [255, 255, 255])):
            bar_x_left = p  # start of the scale bar on the x-axis
        else:
            break

    # from x_ref, to the right on the x
    # until it becomes white, where the scale bar ends
    for p in range(x_ref, max_x):
        if np.any(np.not_equal(img[bar_y][p], [255, 255, 255])):
            bar_x_right = p  # end of the scale bar on the x-axis
        else:
            break

    # size of the scale bar
    bar_size = bar_x_right - bar_x_left

    # using pytesseract (Tesseract OCR for Python) to read text from image
    # the factor is then identified as being any number followed by " nm"
    text = pytesseract.image_to_string(img)
    try:  # tries to read scale factor from image
        factor = int(re.sub(r'[^0-9]', '', re.search(r'[0-9]+ nm', text).group(0)))
    except AttributeError:  # if failed, scale will be 1:1
        factor = bar_size

    return factor / bar_size  # returns scale


# index view of the nanoer app
# any user can submit an image for analysis
# even if they are not logged-in or do not have an account
class IndexView(TemplateView):
    model = Project
    template_name = "nanoer/index.html"
    context_object_name = 'project_list'

    def post(self, request):
        uploaded_file = request.FILES['img']

        # check that the uploaded file is compliant with the required format
        if not hasattr(uploaded_file, 'temporary_file_path') or not uploaded_file.name.lower().endswith(
                ('.tif', '.tiff')):
            return render(request, self.template_name, {'fail': 1})

        path = uploaded_file.temporary_file_path()  # uploaded image
        original_image = cv2.imread(path)  # read uploaded image with OpenCV
        image, legend = crop_image(original_image)  # separate actual from image and legend
        scale = calculate_scale(legend)  # calculate scale from legend

        image = cv2.medianBlur(image, 5)  # blur image to help with edge detection
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # make sure image is gray scale

        try:  # tries to process image
            """
            use OpenCV Hough Circle Transform to detect circles. From the official documentation:
            
            cv.HoughCircles(
              image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
            
            Parameters:
              image=      8-bit, single-channel, grayscale input image.
              circles=    Output vector of found circles. Each vector is encoded as 3 or 4 element 
                          floating-point vector (x,y,radius) or (x,y,radius,votes).
              method=     Detection method, see HoughModes. The available methods are HOUGH_GRADIENT 
                          and HOUGH_GRADIENT_ALT.
              dp=     	  Inverse ratio of the accumulator resolution to the image resolution. For 
                          example, if dp=1 , the accumulator has the same resolution as the input 
                          image. If dp=2 , the accumulator has half as big width and height. For 
                          HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very 
                          circles need to be detected.
              minDist=    Minimum distance between the centers of the detected circles. If the parameter 
                          is too small, multiple neighbor circles may be falsely detected in addition to 
                          a true one. If it is too large, some circles may be missed.
              param1= 	  First method-specific parameter. In case of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT, 
                          it is the higher threshold of the two passed to the Canny edge detector (the 
                          lower one is twice smaller). Note that HOUGH_GRADIENT_ALT uses Scharr algorithm 
                          to compute image derivatives, so the threshold value shough normally be higher, 
                          such as 300 or normally exposed and contrasty images.
              param2= 	  Second method-specific parameter. In case of HOUGH_GRADIENT, it is the 
                          accumulator threshold for the circle centers at the detection stage. 
                          The smaller it is, the more false circles may be detected. Circles, corresponding 
                          to the larger accumulator values, will be returned first. In the case of 
                          HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure. 
                          The closer it to 1, the better shaped circles algorithm selects. In most 
                          cases 0.9 should be fine. If you want get better detection of small circles, 
                          you may decrease it to 0.85, 0.8 or even less. But then also try to limit 
                          the search range [minRadius, maxRadius] to avoid many false circles.
              minRadius=  Minimum circle radius.
              maxRadius=  Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, 
                          HOUGH_GRADIENT returns centers without finding the radius. HOUGH_GRADIENT_ALT 
                          always computes circle radiuses.
            """
            circles = cv2.HoughCircles(image=gray_img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50,
                                       param2=75,
                                       minRadius=25, maxRadius=200)
            circles = np.uint16(np.around(circles))  # round up circle points with numpy

            # back to RGB so that the circles are visible
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            # draw the circle outlines and their center points
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(color_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(color_img, (i[0], i[1]), 2, (0, 0, 255), 3)

            # create pandas dataframe from detected circles
            df = pd.DataFrame(data=circles[0], columns=['center_x', 'center_y', 'diameter_nm'])
            # convert radii to nanometers and multiply by 2 to obtain diameter
            df.diameter_nm *= scale * 2

            # prepare canvas and axes to receive graph
            fig, ax = plt.subplots(tight_layout=True)
            ax2 = plt.twinx(ax)  # one x-axis, two y-axes
            ax.set_title('Size Distribution Histogram')
            if scale == 1:  # if scale is 1:1 diameter is in pixels
                ax.set_xlabel('Diameter (px)')
            else:  # else, diameter is in nanometers
                ax.set_xlabel('Diameter (nm)')
            ax.set_ylabel('Number of Particles')
            ax2.set_ylabel('Density')

            # plot histogram with number of particles on the left y-axis
            sns.histplot(data=df, x='diameter_nm', legend=False, color='skyblue', ax=ax)
            # plot KDE with density on the right y-axis
            sns.kdeplot(data=df, x='diameter_nm', legend=False, color='crimson', ax=ax2)

            tmpdir = tempfile.mkdtemp()  # create temporary directory
            os.mkdir(os.path.join(tmpdir, 'files'))  # makes folder named 'files' inside

            # copy originally uploaded image to the temporary files directory
            shutil.copy2(path, os.path.join(tmpdir, 'files', 'original_image.tif'))
            # saves processed image to temp folder as a PNG file
            cv2.imwrite(os.path.join(tmpdir, 'files', 'processed_image.png'), color_img)
            # saves plotted graph to temp folder as a PNG file
            fig.savefig(os.path.join(tmpdir, 'files', 'size_dist_plot.png'))
            # saves dataframe to temp folder as a CSV file
            df.to_csv(os.path.join(tmpdir, 'files', 'dataframe.csv'))

            # creates a ZIP archive of temp folder
            results = shutil.make_archive(os.path.join(tmpdir, 'u0'), 'zip', tmpdir, 'files')
            tmp_files = os.path.join(tmpdir, 'files')  # path to temp files
            list_dir = sorted(os.listdir(tmp_files))  # list of files within temp files
            # move temp folder to MEDIA_ROOT
            tmpdir = shutil.move(tmpdir, os.path.join(MEDIA_ROOT, 'tmp'))
            os.chmod(tmpdir, 0o770)  # full permission to root and server group, none to others
            # list of individual paths to temp files
            tmp_media_url = [os.path.join(os.sep, MEDIA_URL, tmp_files[1:], file) for file in list_dir]

            context = {'image': tmp_media_url[2], 'chart': tmp_media_url[3], 'results': results, 'tmpdir': tmpdir,
                       'tmp_files': tmp_files}
        except:  # if it fails, for any reason, informs template
            context = {'fail': 2}

        return render(request, self.template_name, context)


# create project view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class CreateView(LoginRequiredMixin, UserPassesTestMixin, generic.CreateView):  # create new poll view
    model = Project
    template_name = 'nanoer/create.html'
    fields = ['title', 'description']

    def post(self, request, **kwargs):
        # instantiates Project model
        project = Project.objects.create(title=request.POST['title'], description=request.POST['description'],
                                         user_id=kwargs['pk'])

        if not project.title:  # provides generic title if user did not provide any
            project.title = f'Project created at {project.created_at}.'
        if not project.description:  # provides generic description if user did not provide any
            project.description = f'This project was created at {project.created_at}. Both title and description were ' \
                                  f'auto-generated and can be changed by editing the project\'s details.'

        project.save()  # saves to database

        # creates folder in the server to store the new project's future files
        os.mkdir(os.path.join(MEDIA_ROOT, str(project.pk)))

        return HttpResponseRedirect(reverse('nanoer:detail', kwargs={'pk': project.pk}))

    def test_func(self):  # logged-in user has to be the owner of the new project
        return self.request.user.id == self.kwargs['pk']


# delete project view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class DeleteView(LoginRequiredMixin, UserPassesTestMixin, generic.DeleteView):
    model = Project
    template_name = 'nanoer/delete.html'

    def get(self, request, *args, **kwargs):
        self.model = Project.objects.get(pk=kwargs['pk'])  # retrieves project from database by id

        # renders generic confirmation of deletion template
        return render(request, self.template_name,
                      {'model': self.model, 'model_classname': self.model.__class__.__name__})

    def test_func(self):  # logged-in user has to be the owner of the project to be deleted
        return self.request.user.id == Project.objects.get(id=self.kwargs['pk']).user_id


# delete analysis view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class AnalysisDeleteView(LoginRequiredMixin, UserPassesTestMixin, generic.DeleteView):
    model = Analysis
    template_name = 'nanoer/delete.html'

    def get(self, request, *args, **kwargs):
        self.model = Analysis.objects.get(pk=kwargs['pk'])  # retrieves analysis from database by id

        # renders generic confirmation of deletion template
        return render(request, self.template_name,
                      {'model': self.model, 'model_classname': self.model.__class__.__name__})

    def test_func(self):  # logged-in user has to be the owner of the project to be deleted
        project_id = Analysis.objects.get(id=self.kwargs['pk']).project_id
        return self.request.user.id == Project.objects.get(id=project_id).user_id


# confirm deletion generic view for specific user
# only logged-in users can access this view
@login_required
def delete_confirm(request, *args, **kwargs):
    # resolves the type of model being deleted and retrieves the correct instance from database by id
    model = getattr(sys.modules[__name__], kwargs['model']).objects.get(pk=kwargs['pk'])
    model.delete()  # delete from database

    # redirection depends on the model type
    if kwargs['model'] == "Project":
        return redirect('accounts:projects', model.user_id)
    else:
        return redirect('nanoer:detail', model.project_id)


# detail project view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class DetailView(LoginRequiredMixin, UserPassesTestMixin, generic.ListView):
    model = Project
    template_name = 'nanoer/detail.html'
    context_object_name = 'context'

    def get_queryset(self):
        project_id = self.kwargs['pk']
        project = Project.objects.get(id=project_id)  # project from database by id
        # list of analyses that belong to this project
        analysis_list = Analysis.objects.filter(project_id=project_id).order_by('-created_at')
        # list of files that belong to those analyses
        file_list = File.objects.filter(analysis_id__in=[a.id for a in analysis_list])

        return {'project': project, 'analysis_list': analysis_list, 'file_list': file_list}

    def test_func(self):  # logged-in user has to be the owner of the project to be accessed
        return self.request.user.id == Project.objects.get(id=self.kwargs['pk']).user_id


# temporary files clean-up method
def temp_clean_up(request):
    print(request.POST['tmpdir'])
    shutil.rmtree(request.POST['tmpdir'])  # remove file tree rooted at tmpdir

    return HttpResponse(status=204)


# get list of projects method for specific user
# only logged-in users can access this view
@login_required
def get_project_list(request):
    # gets list of projects that belong to the user
    project_list = Project.objects.filter(user_id=request.GET['userId']).values().order_by('-created_at')

    # returns it as JSON to the AJAX call
    return JsonResponse(list(project_list), safe=False)


# save to project method for specific user
# only logged-in users can access this view
@login_required
def save_to_project(request):
    project_id = request.POST['projectId']  # project where analysis will be saved

    # instantiates Analysis model
    analysis = Analysis.objects.create(project_id=project_id)

    # assigns temporary generic title and description to analysis
    analysis.title = f'Analysis created at {analysis.created_at}.'
    analysis.description = f'This analysis was created at {analysis.created_at}. Both title and description were ' \
                           f'auto-generated and can be changed by clicking on the pencil icon.'

    # save to database
    analysis.save()

    # path to where media is stored in the server
    uri = os.path.join(MEDIA_ROOT, str(project_id), 'analyses', str(analysis.id), 'files')
    # copy analysis' files from temp folder to storage
    shutil.copytree(os.path.join(MEDIA_ROOT, request.POST['tmpFiles'][1:]), uri)
    # path for serving the media content from storage
    uri = os.path.join(os.sep, MEDIA_URL, str(project_id), 'analyses', str(analysis.id), 'files')

    # instantiates File model for the original file so that it can be
    # used as source_id for the other files of the analysis
    original_image = File.objects.create(title="original_image", uri=os.path.join(uri, 'original_image.tif'),
                                         type="image", analysis_id=analysis.id)
    original_image.save()  # save to database

    # creates and saves the other files of the analysis
    File.objects.create(title="processed_image", uri=os.path.join(uri, 'processed_image.png'), type="image",
                        source_id=original_image.id, analysis_id=analysis.id).save()
    File.objects.create(title="size_dist_plot", uri=os.path.join(uri, 'size_dist_plot.png'), type="image",
                        source_id=original_image.id, analysis_id=analysis.id).save()
    File.objects.create(title="dataframe", uri=os.path.join(uri, 'dataframe.csv'), type="data",
                        source_id=original_image.id, analysis_id=analysis.id).save()

    # returns it as JSON to the AJAX call
    return JsonResponse(f'project/{project_id}', safe=False)


# download analysis method for specific user
# only logged-in users can access this view
@login_required
def download_analysis(request, **kwargs):
    # get ids of associated user, project and analysis
    analysis_id = kwargs['pk']
    project_id = Analysis.objects.get(id=analysis_id).project_id
    user_id = Project.objects.get(id=project_id).user_id

    # path to analysis archive
    uri = os.path.join(MEDIA_ROOT, str(project_id), 'analyses', str(analysis_id))
    # archive title
    archive = f'u{user_id}_p{project_id}_a{analysis_id}'
    # create ZIP archive of files in the analysis
    shutil.make_archive(os.path.join(uri, archive), 'zip', uri, 'files')
    # url for download of the archive
    media_url = os.path.join(os.sep, MEDIA_URL, str(project_id), 'analyses', str(analysis_id), f'{archive}.zip')

    return redirect(media_url)


# download project method for specific user
# only logged-in users can access this view
@login_required
def download_project(request, **kwargs):
    # get ids of associated user and project
    project_id = kwargs['pk']
    user_id = Project.objects.get(id=project_id).user_id

    # path to project archive
    uri = os.path.join(MEDIA_ROOT, str(project_id))
    # archive title
    archive = f'u{user_id}_p{project_id}'

    # creates temp folder to filter out any previously created ZIP archives
    # left inside any analyses in this project
    with tempfile.TemporaryDirectory() as tmpdir:
        # copy file tree rooted at the project's folder ignoring ZIP archives
        shutil.copytree(src=uri, dst=tmpdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.zip'))
        # create ZIP archive of files and folders in the project
        shutil.make_archive(os.path.join(uri, archive), 'zip', tmpdir, 'analyses')

    # url for download of the archive
    media_url = os.path.join(os.sep, MEDIA_URL, str(project_id), f'{archive}.zip')

    return redirect(media_url)


# update project view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class ProjectUpdateView(LoginRequiredMixin, UserPassesTestMixin, generic.UpdateView):
    model = Project
    template_name = 'nanoer/edit_project.html'
    fields = []

    def post(self, request, *args, **kwargs):
        self.model = Project.objects.get(pk=kwargs['pk'])  # retrieves project from database by id
        self.model.title = request.POST['title']  # sets its new title
        self.model.description = request.POST['description']  # sets its new description
        self.model.save()  # saves to database
        # renders account projects view for specific user
        return redirect('accounts:projects', self.model.user_id)

    def test_func(self):  # logged-in user has to be the owner of the project to be updated
        return self.request.user.id == Project.objects.get(id=self.kwargs['pk']).user_id


# update analysis view for specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class AnalysisUpdateView(LoginRequiredMixin, UserPassesTestMixin, generic.UpdateView):
    model = Analysis
    template_name = 'nanoer/edit_analysis.html'
    fields = []

    def post(self, request, *args, **kwargs):
        self.model = Analysis.objects.get(pk=kwargs['pk'])  # retrieves analysis from database by id
        self.model.title = request.POST['title']  # sets its new title
        self.model.description = request.POST['description']  # sets its new description
        self.model.save()  # saves to database
        # renders project details view for specific project
        return redirect('nanoer:detail', self.model.project_id)

    def test_func(self):  # logged-in user has to be the owner of the analysis to be updated
        project_id = Analysis.objects.get(id=self.kwargs['pk']).project_id
        return self.request.user.id == Project.objects.get(id=project_id).user_id
