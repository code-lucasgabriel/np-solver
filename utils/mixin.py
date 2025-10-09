def create_solver(base_class, mixins: list, **kwargs):
    """
    Factory function to dynamically create a solver class and instantiate it.

    This function builds a new class that inherits from the provided mixins
    and the base class, then creates an instance of it.

    Args:
        base_class: The main class to be instantiated (e.g., GA_SCQBF).
        mixins (list): A list of mixin classes to add to the object's behavior.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the dynamically created class.
    """
    # ! The order is important: mixins first to ensure their methods override the base class methods via Python's Method Resolution Order (MRO).
    class_name = f"{base_class.__name__}Dynamic"
    
    dynamic_class = type(class_name, (*mixins, base_class), {})
    
    return dynamic_class(**kwargs)