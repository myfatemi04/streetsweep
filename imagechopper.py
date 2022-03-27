def chop_image(image, x_chops, y_chops):
    bounding_boxes = []
    sections = []

    height, width = image.shape[-2:]

    for x_chop in range(x_chops):
        for y_chop in range(y_chops):
            x_chop_size = int(width / x_chops)
            y_chop_size = int(height / y_chops)
            section = image[:, y_chop * y_chop_size:(y_chop + 1) *
                            y_chop_size, x_chop * x_chop_size:(x_chop + 1) * x_chop_size]

            bounding_boxes.append(
                ((x_chop * x_chop_size, y_chop * y_chop_size),
                 ((x_chop + 1) * x_chop_size, (y_chop + 1) * y_chop_size))
            )

            sections.append(section)

    return bounding_boxes, sections
