import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import time

def detect_dominant_color(image, k=1):
    # Redimensionner l'image pour accélérer le traitement
    image = cv2.resize(image, (100, 100))
    # Conversion en format de pixels pour KMeans
    pixels = image.reshape(-1, 3)
    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    # Retourne la couleur dominante (centre du cluster principal)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color


def color_distance(color1, color2):
    # Calcul de la distance euclidienne entre deux couleurs
    return np.linalg.norm(color1 - color2)


def selection_sort_by_color(dataset, colors):
    # Tri par sélection basé sur la couleur dominante
    for i in range(len(dataset)):
        min_index = i
        for j in range(i + 1, len(dataset)):
            if color_distance(colors[j], [0, 0, 0]) < color_distance(colors[min_index], [0, 0, 0]):
                min_index = j
        dataset[i], dataset[min_index] = dataset[min_index], dataset[i]
        colors[i], colors[min_index] = colors[min_index], colors[i]

# Tri par insertion
def insertion_sort_by_color(dataset, colors):
    for i in range(1, len(dataset)):
        key_data = dataset[i]
        key_color = colors[i]
        j = i - 1
        while j >= 0 and color_distance(colors[j], [0, 0, 0]) > color_distance(key_color, [0, 0, 0]):
            dataset[j + 1] = dataset[j]
            colors[j + 1] = colors[j]
            j -= 1
        dataset[j + 1] = key_data
        colors[j + 1] = key_color


# Tri par fusion
def merge_sort_by_color(dataset, colors):
    if len(dataset) > 1:
        mid = len(dataset) // 2
        left_dataset = dataset[:mid]
        right_dataset = dataset[mid:]
        left_colors = colors[:mid]
        right_colors = colors[mid:]

        merge_sort_by_color(left_dataset, left_colors)
        merge_sort_by_color(right_dataset, right_colors)

        i = j = k = 0
        while i < len(left_dataset) and j < len(right_dataset):
            if color_distance(left_colors[i], [0, 0, 0]) < color_distance(right_colors[j], [0, 0, 0]):
                dataset[k] = left_dataset[i]
                colors[k] = left_colors[i]
                i += 1
            else:
                dataset[k] = right_dataset[j]
                colors[k] = right_colors[j]
                j += 1
            k += 1

        while i < len(left_dataset):
            dataset[k] = left_dataset[i]
            colors[k] = left_colors[i]
            i += 1
            k += 1

        while j < len(right_dataset):
            dataset[k] = right_dataset[j]
            colors[k] = right_colors[j]
            j += 1
            k += 1


def quick_sort_by_color(dataset, colors):
    """Tri rapide des vêtements en fonction de leur couleur dominante."""
    if len(dataset) <= 1:
        return dataset, colors
    else:
        # Pivot basé sur la couleur dominante
        pivot_color = colors[len(colors) // 2]
        pivot_distance = color_distance(pivot_color, [0, 0, 0])

        left_dataset, left_colors = [], []
        right_dataset, right_colors = [], []
        pivot_dataset, pivot_colors = [], []

        for i in range(len(colors)):
            dist = color_distance(colors[i], [0, 0, 0])
            if dist < pivot_distance:
                left_dataset.append(dataset[i])
                left_colors.append(colors[i])
            elif dist > pivot_distance:
                right_dataset.append(dataset[i])
                right_colors.append(colors[i])
            else:
                pivot_dataset.append(dataset[i])
                pivot_colors.append(colors[i])

        # Appel récursif
        sorted_left_dataset, sorted_left_colors = quick_sort_by_color(left_dataset, left_colors)
        sorted_right_dataset, sorted_right_colors = quick_sort_by_color(right_dataset, right_colors)

        # Combiner les résultats
        return (
            sorted_left_dataset + pivot_dataset + sorted_right_dataset,
            sorted_left_colors + pivot_colors + sorted_right_colors,
        )



def process_clothes_images(path, max_images=None):
    dataset = []
    colors = []
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(path, filename))
            if image is not None:
                dominant_color = detect_dominant_color(image)
                dataset.append(filename)
                colors.append(dominant_color)
                count += 1
                if max_images and count >= max_images:
                    break
    return dataset, colors



# Mesure du temps d'exécution
def measure_execution_time(sort_function, dataset, colors):
    dataset_copy = dataset[:]
    colors_copy = colors[:]
    start_time = time.time()
    sort_function(dataset_copy, colors_copy)
    end_time = time.time()
    return end_time - start_time


def plot_complexity(path):
    sizes = [200,400,600,800,1000]  # Tailles simulées de la dataset
    insertion_times = []
    merge_times = []
    selection_times = []
    quick_times = []

    for size in sizes:
        dataset, colors = process_clothes_images(path, max_images=size)

        # Temps pour tri par insertion
        insertion_times.append(measure_execution_time(insertion_sort_by_color, dataset, colors))

        # Temps pour tri par fusion
        merge_times.append(measure_execution_time(merge_sort_by_color, dataset, colors))

        # Temps pour tri par sélection
        selection_times.append(measure_execution_time(selection_sort_by_color, dataset, colors))

        # Temps pour tri rapide
        quick_times.append(measure_execution_time(quick_sort_by_color, dataset, colors))

    # Tracer les courbes
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, insertion_times, marker='o', label="trie par Insertion", color="blue")
    plt.plot(sizes, merge_times, marker='o', label="trie par fusion", color="green")
    plt.plot(sizes, selection_times, marker='o', label="Selection Sort", color="red")
    plt.plot(sizes, quick_times, marker='o', label="trie rapide", color="yellow")
    plt.title("Comparaison des complexités des algorithmes")
    plt.xlabel("Taille de la dataset")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.legend()
    plt.grid(True)
    plt.show()





# Chemin du dossier des images de vêtements
path = "C:/Users/HP/Desktop/algorithmique avancee tp/Dataset/BD/clothes"

# Appliquer le tri
sorted_clothes = process_clothes_images(path)

# Afficher le résultat trié
#for filename in sorted_clothes:
    #print(filename)
plot_complexity(path)