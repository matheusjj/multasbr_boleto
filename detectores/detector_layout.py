import cv2
import layoutparser as lp
import matplotlib.pyplot as plt


class DetectorLayout:
    def __init__(self):
        None

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model = lp.Detectron2LayoutModel(
                    # PublayNet
                    # config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    # config_path='lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config',
                    # config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                    # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    # label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})

                    # TableBank
                    config_path='lp://TableBank/faster_rcnn_R_101_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Table"})

                    # HJDataset
                    # config_path='lp://HJDataset/faster_rcnn_R_50_FPN_3x/config',
                    # config_path='lp://HJDataset/mask_rcnn_R_50_FPN_3x/config',
                    # config_path='lp://HJDataset/retinanet_R_50_FPN_3x/config',
                    # label_map={1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title",
                    #            6: "Subtitle", 7: "Other"})
        layout = model.detect(img)
        lp.draw_box(img, layout, box_width=3)
        for lay in layout:
            cv2.rectangle(img, (int(lay.block.x_1), int(lay.block.y_1)),
                          (int(lay.block.x_2), int(lay.block.y_2)), (255, 0, 0), 1)
            cv2.putText(img, lay.type, (int(lay.block.x_1), int(lay.block.y_1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('a', img)
        cv2.waitKey(0)
        #
        # text_blocks = lp.Layout([b for b in layout if b.type == 'Text'])
        # table_blocks = lp.Layout([b for b in layout if b.type == 'Table'])
        # text_blocks = lp.Layout([b for b in text_blocks \
        #                          if not any(b.is_in(b_fig) for b_fig in table_blocks)])
        #
        # h, w = img.shape[:2]
        #
        # left_interval = lp.Interval(0, w / 2 * 1.05, axis='x').put_on_canvas(img)
        #
        # left_blocks = text_blocks.filter_by(left_interval, center=True)
        # left_blocks.sort(key=lambda b: b.coordinates[1])
        #
        # right_blocks = [b for b in text_blocks if b not in left_blocks]
        # right_blocks.sort(key=lambda b: b.coordinates[1])
        #
        # # And finally combine the two list and add the index
        # # according to the order
        # text_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])
        # lp.draw_box(img, text_blocks,
        #             box_width=3,
        #             show_element_id=True)
        #
        # # lp.draw_text(img, layout, font_size=12, with_box_on_text=True, text_box_width=1)
