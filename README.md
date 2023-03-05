# at³: an automated size distribution calculator for TEM-based nanoparticle scans

**Dorset College Dublin**  
**BSc in Science in Computing & Multimedia**  
**Research Project / Third Year Project - BSC30922**  
**Year 3, Semester 2**  
**Final Project**

**Coordinator name:** Marie Keary  
**Coordinator email:** marie.keary@dorset.ie  

**Supervisor name:** Albert Martinez  
**Supervisor email:** alberto.martinez@dorset.ie 

**Student Name:** Mateus Fonseca Campos  
**Student Number:** 24088  
**Student Email:** 24088@student.dorset-college.ie

**Submission date:** 5 March 2023

This repository contains a Django web app developed as part of my Research Project/Third Year Project at Dorset College BSc in Computing, Year 3, Semester 2.

## Part 1: Requirements and Setup

**Framework:** this project requires installation of both [Python](https://www.python.org/downloads/) and [Django](https://www.djangoproject.com/download/).

**Database engine:** it also requires access to a database through an user with enough privileges to create new tables and manipulate the data therein ([see docs](https://docs.djangoproject.com/en/4.1/ref/databases/)). The project is currently set to work with [MySQL](https://dev.mysql.com/downloads/), but switching to another supported backend should be an easy fix.

**Migration:** it is necessary to tell Django to create the tables in the database from the models defined in the project. On a terminal window at the project's root directory:

    python manage.py makemigrations
    python manage.py migrate

**Fixture:** once the tables are created, it is possible to populate them with predefined data to start playing with the app right away. The project contains the file */fixtures/data.json*, which is a fixture, a JSON object that tells Django what data to use to populate the tables in the database. On the terminal:

    python manage.py loaddata data.json

**Packages:** since the application relies on a number of third-party packages, their installation can be automated. The project contains the file /requirements.txt, that cen be read by pip to have all packages installed at once. On the terminal:

    pip install -r requirements.txt

## Part 2: Background

The aim of this project was to develop a web application that makes use of computer vision to assist nanotechnologists in calculating the size distribution of nanoparticle populations by automating the analysis of images obtained via TEM and reducing human intervention, thus quickening the characterization process and increasing its reliability by removing an error-prone step.

You can read more about **[nanoparticles](https://doi.org/10.1002/9783527648122.ch1)**, **[TEM characterization](https://doi.org/10.1039/9781782621867-00001)** and **[computer vision](https://doi.org/10.1007/978-0-387-31439-6)**.

## Part 3: Breakdown

This project was developed based on the web framework Django and its MVT design pattern. The following scheme explains the organization of its main components:

- **1. Models**
  - ```mermaid
    erDiagram
      PROJECT {
          int id PK "The id of the project"
          string title "The title of the project"
          string description "The description of the project"
          datetime created_at "The timestamp of the project's creation"
          datetime updated_at "The timestamp of the project's last update"
          int user_id FK "The id of the user that owns the project"
      }
    ```
  - ```mermaid
    erDiagram
      ANALYSIS {
          int id PK "The id of the analysis"
          string title "The title of the analysis"
          string description "The description of the analysis"
          datetime created_at "The timestamp of the analysis' creation"
          int project_id FK "The id of the project that owns the analysis"
      }
    ```
  - ```mermaid
    erDiagram
      FILE {
          int id PK "The id of the file"
          string title "The title of the file"
          string uri "The file's location on the server"
          string type "The file's type ('image' or 'data')"
          int source_id FK "The id of the file on which the analysis was based"
          int analysis_id FK "The id of the analysis that owns the file"
      }
    ```

- **2. Views**
  - **accounts.SignUpView:** create new user account.
  - **accounts.DetailView:** display user account details.
  - **accounts.DeleteView:** delete user account.
  - **accounts.UpdateEmailView:** update user account details.
  - **accounts.ProjectsView:** list user projects.
  - **nanoer.IndexView:** run analysis.
  - **nanoer.CreateView:** create new project.
  - **nanoer.DeleteView:** delete project.
  - **nanoer.AnalysisDeleteView:** delete analysis.
  - **nanoer.DetailView:** display project details.
  - **nanoer.ProjectUpdateView:** update project details.
  - **nanoer.AnalysisUpdateView:** update analysis details.
  - **nanoer.crop_image:** crop image for analysis.
  - **nanoer.calculate_scale:** read image's legend to obtain scaling factor.
  - **nanoer.delete_confirm:** confirm object deletion.
  - **nanoer.temp_clean_up:** delete temporary files generated during analysis.
  - **nanoer.get_project_list:** get list of user projects.
  - **nanoer.save_to_project:** save analysis' results to an existing project.
  - **nanoer.download_analysis:** download analysis.
  - **nanoer.download_project:** download project.

- **3. Templates**
  - **./about.html:** about page.
  - **./base.html:** shared header, navbar and footer.
  - **./home.html:** home page.
  - **./registration/login.html:** login page.
  - **./registration/password_change_done.html:** password change confirmation.
  - **./registration/password_change_form.html:** password change request.
  - **./registration/password_reset_complete.html:** password reset completed.
  - **./registration/password_reset_confirm.html:** new password.
  - **./registration/password_reset_done.html:** password reset email notification.
  - **./registration/password_reset_form.html:** password reset request.
  - **./registration/signup.html:** sign up page.
  - **./accounts/change_email.html:** change email request.
  - **./accounts/delete.html:** delete account request.
  - **./accounts/delete_confirmation.html:** delete account confirmation.
  - **./accounts/detail.html:** account details.
  - **./accounts/projects.html:** user projects list.
  - **./nanoer/create.html:** create new project.
  - **./nanoer/delete.html:** delete project or analysis.
  - **./nanoer/detail.html:** project details.
  - **./nanoer/edit_analysis.html:** edit analysis details.
  - **./nanoer/edit_project.html:** edit project details.
  - **./nanoer/index.html:** submit file for analysis.
  - **./nanoer/save_to_project.html:** save analysis' results to project.

## Part 4: References

This project runs on some juicy <span style="color: green">**venom**</span>:

- **[OpenCV](https://opencv.org/)** (**[opencv-python](https://pypi.org/project/opencv-python/)**)
- **[NumPy](https://numpy.org/)**
- **[Pandas](https://pandas.pydata.org/)**
- **[Seaborn](https://seaborn.pydata.org/)**
- **[Matplotlib](https://matplotlib.org/)**
- **[Tesseract OCR](https://tesseract-ocr.github.io/)** (**[pytesseract](https://pypi.org/project/pytesseract/)**)

Held together by **[Django](https://www.djangoproject.com/)** and **[Python](https://www.python.org/)**.

## Part 5: Copyright Disclaimer

This project may feature content that is copyright protected. Please, keep in mind that this is a student's project and has no commercial purpose whatsoever. Having said that, if you are the owner of any content featured here and would like for it to be removed, please, contact me and I will do so promptly.

Thank you very much,  
Mateus Campos.
