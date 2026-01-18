from filters.base_3d_filter import Base3DFilter

class HockeyMaskFilter(Base3DFilter):
    def __init__(self):
        super().__init__(
            model_path="assets/Hockey_Mask.obj",
            color=(200, 200, 255),
            alpha=0.85
        )