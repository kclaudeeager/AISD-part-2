from base_iris_lab1 import load_local, new_model, score

dataset = load_local()
print(dataset)
# should return dataset index 0 for the first dataset

model = new_model(dataset)
print(model)
# should return model index 0 for the first model instance

# you can also experiment with providing custom values, e.g.:
features = [
    1000,  # elevation
    1,  # soil_type
    8.5,  # sepal_length
    5.3,  # sepal_width
    16,  # petal_length
    24,  # petal_width
    45.25,  # sepal_area
    384,  # petal_area
    1.6,  # sepal_aspect_ratio
    0.67,  # petal_aspect_ratio
    0.53,  # sepal_to_petal_length_ratio
    0.22,  # sepal_to_petal_width_ratio
    -7.5,  # sepal_petal_length_diff
    -18.7,  # sepal_petal_width_diff
    0.2,  # petal_curvature_mm
    100,  # petal_texture_trichomes_per_mm2
    200,  # leaf_area_cm2
    6.7,  # sepal_area_sqrt
    19.6,  # petal_area_sqrt
    0.12  # area_ratios
]

score_result = score(model, features)
print(score_result)