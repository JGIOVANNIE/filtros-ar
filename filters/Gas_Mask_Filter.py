from filters.base_3d_filter import Base3DFilter

class GasMaskFilter(Base3DFilter):
    def __init__(self):
        super().__init__(
            model_path="assets/Gas_Mask.obj",
            color=(200, 200, 255),
            alpha=0.85
            
        )
