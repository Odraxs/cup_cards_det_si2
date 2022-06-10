import cv2

class Recorta:

    def recortar(self, ruta, image, contours, category, num):
        idNum = category
        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if w > 50 and h > 50:
                # Redimenciona
                new_img = image[y:y + h, x:x + w]
                cv2.imwrite(ruta + str(idNum) + '_' + str(num) + '.jpg', new_img)
                idNum = idNum + 1
        return num