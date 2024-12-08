def match_apparel(model, img1, img2):
    encoded_img1 = model(img1)
    encoded_img2 = model(img2)
    # compatibility_score = compatible_check(encoded_img1, encoded_img2)
    # return compatibility_score