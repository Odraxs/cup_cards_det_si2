import cv2
from  pruebas import  predecir
class Cut:
    # function to crop image
    # Parameters: image to crop, contour, and the image number
    def crop(image, contours, num, bordes):
        pru = image
        new_img = bordes
        idNum = num

        # cycles through all the contours to crop all
        for c in contours:
            area = cv2.contourArea(c)
            if area == 0:
                break
            # creates an approximate rectangle around contour
            x, y, w, h = cv2.boundingRect(c)
            # Only crop decently large rectangles
            # cv2.imshow("Bordes", imagen)
            # if( w>100 and h>100):
            #     print('w',w)
            #     print('h',h)
            if (w > 370 and h > 180) or (h > 370 and w > 180):
                # print("w es %s y h es %s" %(w,h))
                # cv2.drawContours(pru, [c], 0, (0, 255, 255), 2)
                #cv2.imshow("Recorte", image)
                # print("Oprima c para recortar")
                new_img = bordes[y:y + h, x:x + w]

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    # pulls crop out of the image based on dimensions
                    idNum += 1
                    # writes the new file in the Crops folder
                    cv2.imwrite('Crops/new_test' + '1_' + str(idNum) +
                                '.jpg', image)
                    print("Se tomó el recorte", idNum)

        # returns a number incremented up for the next file name
        return idNum, new_img

    def crop2(image, contours, num, bordes,gris):
        pru = image
        new_img = bordes
        idNum = num
        # cycles through all the contours to crop all
        for c in contours:
            area = cv2.contourArea(c)
            if area == 0:
                break
            # creates an approximate rectangle around contour
            x, y, w, h = cv2.boundingRect(c)
            # Only crop decently large rectangles
            # cv2.imshow("Bordes", imagen)
            # if( w>100 and h>100):
            #     print('w',w)
            #     print('h',h)
            if (w > 370 and h > 180) or (h > 370 and w > 180):
                # print("w es %s y h es %s" %(w,h))
                # cv2.drawContours(pru, [c], 0, (0, 255, 255), 2)
                #cv2.imshow("Recorte", image)
                # print("Oprima c para recortar")
                new_img = bordes[y:y + h, x:x + w]
                gris = gris[y:y + h, x:x + w]
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    # pulls crop out of the image based on dimensions
                    idNum += 1
                    # writes the new file in the Crops folder
                    cv2.imwrite('Crops/Regla/b_n/' + '1_' + str(idNum) +
                                '.jpg', gris)
                    cv2.imwrite('Crops/Regla/bordes/' + '1_' + str(idNum) +
                                '.jpg', new_img)
                    # predecir('Crops/new_test/' + '1_' + str(idNum) +
                    #             '.jpg',gris,c)

                    print("Se tomó el recorte", idNum)

        # returns a number incremented up for the next file name
        return idNum, new_img