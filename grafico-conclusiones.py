import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

# Datos de la tabla proporcionada
sizes = [2, 4, 8, 16, 32, 64]
overhead = [1.70, 0.84, 0.90, 0.71, 0.74, 1.49]
error_pct = [69.8, -16.5, -9.9, -28.8, -26.3, 49.3]

# Crear figura
plt.figure(figsize=(10, 6))

# Definir los límites de los regímenes
regime_1_end = 2.5    # Límite entre régimen subóptimo y óptimo
regime_2_end = 40     # Límite entre régimen óptimo y degradado

# Añadir áreas de color para los regímenes
ax = plt.gca()

# Régimen 1: Subóptimo (n=2)
ax.add_patch(patches.Rectangle((0, 0), regime_1_end, 2.0, 
                              facecolor='#FFCCCC', alpha=0.3, 
                              label='Régimen Subóptimo'))

# Régimen 2: Óptimo (n=4-32)
ax.add_patch(patches.Rectangle((regime_1_end, 0), regime_2_end-regime_1_end, 2.0, 
                              facecolor='#CCFFCC', alpha=0.3, 
                              label='Régimen Óptimo'))

# Régimen 3: Degradado (n≥64)
ax.add_patch(patches.Rectangle((regime_2_end, 0), 100, 2.0, 
                              facecolor='#FFCCCC', alpha=0.3, 
                              label='Régimen Degradado'))

# Graficar los puntos de overhead
plt.plot(sizes, overhead, 'o-', color='darkblue', linewidth=2, markersize=8, label='Overhead')

# Línea de referencia en overhead=1
plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Umbral teórico (overhead=1)')

# Etiquetas de los puntos
for i, (n, oh) in enumerate(zip(sizes, overhead)):
    # Posición vertical de la etiqueta (ajustada para evitar solapamiento)
    y_pos = oh + 0.15 if i < len(sizes)-1 else oh - 0.25
    # Color de la etiqueta según el error
    text_color = 'green' if error_pct[i] < 0 else 'red'
    # Etiqueta
    plt.text(n, y_pos, f'n={n}\nOH={oh:.2f}', 
             ha='center', va='bottom' if error_pct[i] < 0 else 'top',
             color=text_color, fontweight='bold')

# Etiquetas de los regímenes
plt.text((regime_1_end-0)/2, 1.8, 'Régimen\nSubóptimo', 
         ha='center', va='center', fontsize=12, fontweight='bold')
plt.text((regime_1_end + regime_2_end)/2, 1.8, 'Régimen\nÓptimo', 
         ha='center', va='center', fontsize=12, fontweight='bold')
plt.text((regime_2_end + 100)/2, 1.8, 'Régimen\nDegradado', 
         ha='center', va='center', fontsize=12, fontweight='bold')

# Líneas divisorias entre regímenes
plt.axvline(x=regime_1_end, color='gray', linestyle='-', alpha=0.5)
plt.axvline(x=regime_2_end, color='gray', linestyle='-', alpha=0.5)

# Etiquetas y formato
plt.title('Figura 5: Regímenes de Comportamiento del Algoritmo de Strassen', fontsize=14, fontweight='bold')
plt.xlabel('Tamaño de la matriz (n)', fontsize=12)
plt.ylabel('Overhead (Tiempo medido / Tiempo teórico)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log')  # Escala logarítmica para n (mejor visualización de órdenes de magnitud)
plt.ylim(0, 2.0)
plt.xlim(1.8, 80)

# Añadir leyenda personalizada
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkblue', marker='o', linestyle='-', 
           label='Overhead medido'),
    Line2D([0], [0], color='r', linestyle='--', 
           label='Umbral teórico (overhead=1)'),
    patches.Patch(facecolor='#FFCCCC', alpha=0.3, 
                 label='Rendimiento subóptimo (overhead > 1)'),
    patches.Patch(facecolor='#CCFFCC', alpha=0.3, 
                 label='Rendimiento óptimo (overhead < 1)')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Añadir anotaciones explicativas
plt.annotate('Máximo beneficio\nen n=16', 
             xy=(16, 0.71), xytext=(20, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, backgroundcolor='white')

plt.annotate('Transición crítica\na n=64', 
             xy=(64, 1.49), xytext=(45, 1.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=10, backgroundcolor='white')

# Ajustes finales
plt.tight_layout()
plt.savefig('regimenes_comportamiento.png', dpi=300, bbox_inches='tight')
print("✅ Figura generada y guardada como 'figura_regimenes_comportamiento.png'")

plt.show()