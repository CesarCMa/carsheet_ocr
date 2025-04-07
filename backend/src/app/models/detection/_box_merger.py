from typing import Optional
import numpy as np


class BoxMerger:
    def __init__(
        self,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 1.0,
        add_margin: float = 0.05,
        min_size: Optional[int] = None,
    ):
        """Class to merge closer text boxes into a single one.

        Args:
            slope_ths (float, optional): Threshold to determine wether a box is horizontal or not.
                Defaults to 0.1.
            ycenter_ths (float, optional): Threshold used during horizontal merge to determine
                whether or not to merge two boxes. Defaults to 0.5.
            height_ths (float, optional): Threshold used during vertical merge. Defaults to 0.5.
            width_ths (float, optional): Threshold used during vertical merge. Defaults to 1.0.
            add_margin (float, optional): Margin to be added to the final boxes. Defaults to 0.05.
        """
        self.slope_ths = slope_ths
        self.ycenter_ths = ycenter_ths
        self.height_ths = height_ths
        self.width_ths = width_ths
        self.add_margin = add_margin
        self.input_shape = None
        self.min_size = min_size

    def merge_text_boxes(self, text_boxes: np.ndarray) -> tuple:
        """Grouping text boxes into lines.

        Args:
            text_boxes (_type_): List containing coordinates of all vertexes of all text boxes.
                Vertices order should be clockwise and (0, 0) is at top-left.

        Returns:
            _type_: Tuple containing two lists: one with merged text boxes and another with free
                text boxes. Note: result is area of text boxes, not coordinates of the vertices.
                This means that each element in the list should be a list with four elements:
                [x_min, x_max, y_min, y_max].
        """
        merged_horizontal_boxes = []

        self.input_shape = text_boxes.shape
        text_boxes = _prepare_boxes_coordinates(text_boxes, self.input_shape)

        horizontal_list, free_boxes = self._filter_horizontal_boxes(text_boxes)
        combined_horizontal = self._merge_boxes_horizontally(horizontal_list)

        # here two things are happening: adding margin and merging boxes vertically.
        # Maybe this could be separated into two different methods.
        for boxes in combined_horizontal:
            if len(boxes) == 1:
                box = boxes[0]

                # Margin is add_margin times min of width and height
                margin = int(self.add_margin * min(box[1] - box[0], box[5]))
                merged_horizontal_boxes.append(
                    [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
                )
            else:
                # Sort boxes by x_min
                boxes = sorted(boxes, key=lambda item: item[0])

                merged_box, new_box = [], []
                for box in boxes:
                    if len(new_box) == 0:
                        b_height = [box[5]]
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if (
                            abs(np.mean(b_height) - box[5])
                            < self.height_ths * np.mean(b_height)
                        ) and ((box[0] - x_max) < self.width_ths * (box[3] - box[2])):
                            b_height.append(box[5])
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            b_height = [box[5]]
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
                if len(new_box) > 0:
                    merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1:
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]

                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        margin = int(self.add_margin * (min(box_width, box_height)))

                        merged_horizontal_boxes.append(
                            [
                                x_min - margin,
                                x_max + margin,
                                y_min - margin,
                                y_max + margin,
                            ]
                        )
                    else:
                        box = mbox[0]

                        box_width = box[1] - box[0]
                        box_height = box[3] - box[2]
                        margin = int(self.add_margin * (min(box_width, box_height)))

                        merged_horizontal_boxes.append(
                            [
                                box[0] - margin,
                                box[1] + margin,
                                box[2] - margin,
                                box[3] + margin,
                            ]
                        )

        if self.min_size:
            merged_horizontal_boxes, free_boxes = self._filter_boxes_by_size(
                merged_horizontal_boxes, free_boxes
            )

        # Is there any way to return just a single list?  -> check old code
        return merged_horizontal_boxes, free_boxes

    def _filter_horizontal_boxes(self, text_boxes: np.ndarray) -> tuple:
        """Generate two different lists of text boxes: one with horizontal boxes and another with
        free boxes.

        Horizontal boxes are those with a slope less than the threshold, while free boxes are those
        with a slope greater than the threshold. The slope used to determine if a box is horizontal
        is the maximum between the slope of the upper side and the slope of the lower side.

        Args:
            text_boxes (np.ndarray): Array containing the coordinates of the text boxes.

        Returns:
            tuple: A tuple containing two lists: one with horizontal boxes and another with free
                boxes. The horizontal boxes are represented by a list of lists, where each list
                contains the following elements: [x_min, x_max, y_min, y_max, y_center, height].
        """
        horizontal_boxes, free_boxes = [], []

        for poly in text_boxes:
            slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
            slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))

            if max(abs(slope_up), abs(slope_down)) < self.slope_ths:
                x_max = max([poly[0], poly[2], poly[4], poly[6]])
                x_min = min([poly[0], poly[2], poly[4], poly[6]])
                y_max = max([poly[1], poly[3], poly[5], poly[7]])
                y_min = min([poly[1], poly[3], poly[5], poly[7]])

                horizontal_boxes.append(
                    [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
                )
            else:
                height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
                width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
                margin = int(1.44 * self.add_margin * min(width, height))

                theta13 = abs(
                    np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4])))
                )
                theta24 = abs(
                    np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6])))
                )

                x1 = poly[0] - np.cos(theta13) * margin
                y1 = poly[1] - np.sin(theta13) * margin
                x2 = poly[2] + np.cos(theta24) * margin
                y2 = poly[3] - np.sin(theta24) * margin
                x3 = poly[4] + np.cos(theta13) * margin
                y3 = poly[5] + np.sin(theta13) * margin
                x4 = poly[6] - np.cos(theta24) * margin
                y4 = poly[7] + np.sin(theta24) * margin

                free_boxes.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        return horizontal_boxes, free_boxes

    def _merge_boxes_horizontally(self, horizontal_list: list) -> list:
        """Merged horizontal boxes based o the following strategy:
            1. Sort the boxes by the y_center.
            2. Start from the first box, and compare the y_center of the next box with the y_center
                of the current box. If the difference is less than the threshold (`ycenter_ths`
                times the height of the first box), add the box to the current box.
            3. If the difference is greater than the threshold, select the next box as the current
                box and start again.

            If more than 2 boxes meet the criteria then we take mean of distance between y_center
            and mean of height of boxes as threshold.

        Args:
            horizontal_list (list): List containing horizontal boxes data: [x_min, x_max, y_min
                y_max, y_center, height].

        Returns:
            list: List containing merged horizontal boxes (same format as input).
        """
        combined_list = []
        new_box = []

        # Sort the boxes by the y_center
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

        for poly in horizontal_list:
            if len(new_box) == 0:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                new_box.append(poly)
            else:
                if abs(np.mean(b_ycenter) - poly[4]) < self.ycenter_ths * np.mean(
                    b_height
                ):
                    b_height.append(poly[5])
                    b_ycenter.append(poly[4])
                    new_box.append(poly)
                else:
                    b_height = [poly[5]]
                    b_ycenter = [poly[4]]
                    combined_list.append(new_box)
                    new_box = [poly]
        combined_list.append(new_box)

        return combined_list

    def _filter_boxes_by_size(
        self, horizontal_boxes: list, free_boxes: list
    ) -> tuple[list, list]:
        horizontal_boxes = [
            i for i in horizontal_boxes if max(i[1] - i[0], i[3] - i[2]) > self.min_size
        ]
        free_boxes = [
            i
            for i in free_boxes
            if max(_min_max_diff([c[0] for c in i]), _min_max_diff([c[1] for c in i]))
            > self.min_size
        ]
        return horizontal_boxes, free_boxes


def _prepare_boxes_coordinates(
    text_boxes: np.ndarray, input_shape: tuple
) -> np.ndarray:
    """Check the shape of the text boxes and return the correct shape, which is an array with
    vertex coordinates in the followign order: [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    if input_shape[1] == 4 and input_shape[2] == 2:
        return text_boxes.reshape(input_shape[0], -1)
    elif input_shape[1] == 8:
        return text_boxes
    else:
        raise ValueError(
            "Invalid shape of text boxes. It should be (n, 4, 2) or (n, 8)."
        )


def _min_max_diff(input_list: list) -> int:
    """Calculate the difference between the maximum and minimum values in a list.

    Args:
        input_list (list): List of integers.

    Returns:
        int: Difference between the maximum and minimum values in the list.
    """
    return max(input_list) - min(input_list)
