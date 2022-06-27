#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# src » main.py
# ==============================================
# @Author    : Muhammad Wafi <mwafi@mwprolabs.com>
# @Support   : [https://mwprolabs.com]
# @Created   : 19-09-2019
# @Modified  : 27-06-2022 12:07:57 pm
# ----------------------------------------------
# @Copyright (c) 2022 MWprolabs https://mwprolabs.com
#
###


import os
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LeafMorphology:
    def __init__(self):
        self.datasets = glob.glob("dataset/*.jpg")
        self.results = pd.DataFrame(
            columns=[
                "Image Name",
                "AspectRatio",
                "FormFactor",
                "Rectangularity",
                "Narrow Factor",
                "RatioOfDiameter",
                "RatioPLPW",
                "ContourLength",
            ]
        )

    # Plotting multiple images in one figure matplotlib
    def show_images(self, images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert (titles is None) or (len(images) == len(titles))
        n_images = len(images)
        if titles is None:
            titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title, fontdict={"fontsize": 12, "fontweight": "medium"})
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")
        plt.show()

    # Get ASPECT RATIO from image contour
    def get_aspect_ratio(self, length, width):
        aspect_ratio = float(width) / length
        return aspect_ratio

    # Get FORM FACTOR from image contour
    def get_form_factor(self, area, perimeter):
        form_factor = ((4 * np.pi) * area) / ((perimeter) ** 2)
        return form_factor

    # Get RECTANGULARITY from image contour
    def get_rectangularity(self, length, width, area):
        rectangularity = (width * length) / area
        return rectangularity

    # Get NARROW FACTOR from image contour
    def get_narrow_factor(self, diameter, length):
        narrow_factor = diameter / length
        return narrow_factor

    # Get EQUIVALENT DIAMETER
    def get_eq_diameter(self, area):
        eq_diameter = np.sqrt(4 * area / np.pi)
        return eq_diameter

    # Get PERIMETER RATIO OF DIAMETER
    def get_diameter_ratio(self, perimeter, diameter):
        p_diameter_ratio = perimeter / diameter
        return p_diameter_ratio

    # Get PERIMETER RATIO OF PHYSIOLOGICAL LENGTH AND WIDTH
    def get_ratio_plpw(self, perimeter, length, width):
        p_ratio_plpw = perimeter / (width + length)
        return p_ratio_plpw

    # Save results to excel or csv
    def save_results(self, data):
        try:
            data.to_excel(
                "./results/MorphologyFeature.xlsx",
                encoding="utf-8",
                index_label="No",
                header=True,
            )
            print("File has been saved successfully")
            return True
        except IOError:
            print("Cannot export to csv!")
            return False

    # Get Measurements Parameters from image
    def measure_morph(self):
        titles = []
        imgs = []
        for img in self.datasets:
            path, filename = os.path.split(img)
            # read image
            main_img = cv2.imread(img)

            # -- original img append
            imgs.append(main_img)
            titles.append("Original Image")

            # transform to rgb and grayscale image
            rgb_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            grayscale_img = gray_img

            # -- append grayscale imgs
            imgs.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY))
            titles.append("Grayscale Image")

            # blur image using gaussian filter
            blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

            # -- append gaussian imgs
            imgs.append(blur_img)
            titles.append("Gausssian Blur")

            # Adaptive image thresholding using Otsu's thresholding method
            ret_otsu, img_bw_otsu = cv2.threshold(
                blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            _, otsu = cv2.threshold(
                blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # -- append gaussian imgs
            imgs.append(otsu)
            titles.append("Thresholding Image")

            # Boundary extraction using contours
            contours, hierarchy = cv2.findContours(
                img_bw_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            plottedContour = cv2.drawContours(gray_img, contours, -1, (0, 255, 0), 3)

            # -- append boudary extraction imgs
            imgs.append(cv2.drawContours(gray_img, contours, -1, (0, 255, 0), 3))
            titles.append("Contour Extraction")

            print()
            print("Image {} details: ".format(filename))
            print(10 * "---")

            # Image contour
            img_contour = contours[0]
            print("Contour length: {}".format(len(img_contour)))

            # Find centroid using "Image Moments" in cv2 from contours image
            img_moments = cv2.moments(img_contour)

            # Get image contour area
            img_contour_area = cv2.contourArea(img_contour)
            print("Image contour area: {}".format(img_contour_area))

            # Get image "Perimeter"
            perimeter = cv2.arcLength(img_contour, True)
            print("Perimeter: {}".format(perimeter))

            # Fitting image best-fit rectangle and ellipse
            # -- Rectangle Area
            rect = cv2.minAreaRect(img_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img_rect_area = cv2.drawContours(img_bw_otsu, [box], 0, (255, 255, 255), 1)

            # -- Ellipse Area
            ellipse = cv2.fitEllipse(img_contour)
            img_ellipse_area = cv2.ellipse(img_bw_otsu, ellipse, (255, 255, 255), 1)

            # ASPECT RATIO
            x, y, w, h = cv2.boundingRect(img_contour)
            aspect_ratio = self.get_aspect_ratio(h, w)
            print("Aspect Ratio: {}".format(aspect_ratio))

            # RECTANGULARITY
            rectangularity = self.get_rectangularity(h, w, img_contour_area)
            print("Rectangularity: {}".format(rectangularity))

            # FORM FACTOR
            form_factor = self.get_form_factor(img_contour_area, perimeter)
            print("Form Factor: {}".format(form_factor))

            # EQUIVALENT DIAMETER
            equi_diameter = self.get_eq_diameter(img_contour_area)
            print("Equivalent Diameter: {}".format(equi_diameter))

            # Get NARROW FACTOR from image contour area
            narrow_factor = self.get_narrow_factor(equi_diameter, h)
            print("Narrow Factor: {}".format(narrow_factor))

            # Get PERIMETER RATIO OF DIAMETER
            p_diameter_ratio = self.get_diameter_ratio(perimeter, equi_diameter)
            print("Perimeter Ratio of Diameter: {}".format(narrow_factor))

            # Get PERIMETER RATIO OF PHYSIOLOGICAL LENGTH AND PHYSIOLOGICAL WIDTH
            p_ratio_plpw = self.get_ratio_plpw(perimeter, h, w)
            print(
                "Perimeter Ratio of Physiological Length and Width: {}".format(
                    p_ratio_plpw
                )
            )

            # Append images to lists, so can be plotted
            imgs.append(img_ellipse_area)
            titles.append("Best-fit rect and ellipse")

            # save morphology features to pandas
            self.results = self.results.append(
                {
                    "Image Name": filename,
                    "AspectRatio": aspect_ratio,
                    "FormFactor": form_factor,
                    "Rectangularity": rectangularity,
                    "Narrow Factor": narrow_factor,
                    "RatioOfDiameter": p_diameter_ratio,
                    "RatioPLPW": p_ratio_plpw,
                    "ContourLength": len(img_contour),
                },
                ignore_index=True,
            )

        print()

        # Save to xlsx
        self.save_results(self.results)
        # Plot images
        self.show_images(images=imgs, cols=2, titles=titles)

        return "Success..."


if __name__ == "__main__":
    lm = LeafMorphology()
    print(lm.measure_morph())
