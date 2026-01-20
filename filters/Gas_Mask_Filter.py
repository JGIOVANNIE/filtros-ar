from filters.base_3d_filter import Base3DFilter

class GasMaskFilter(Base3DFilter):
    def __init__(self):
        super().__init__(
            model_path="assets/Gas_Mask.obj",
            alpha=0.95
        )
