import numpy as np
import matplotlib.pyplot as plt

# Configure style to match the presentation
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Color palette matching your presentation
primary_colors = ['#0D1826', '#1A2A40', '#4C6173', '#C5CCD9']
accent_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#90EE90', '#FFB6C1']

# Graph 1: Global Mental Health Disorders
def create_global_disorders_chart():
    plt.figure(figsize=(10, 8))
    disorders = ['Depression', 'Anxiety Disorders']
    cases_millions = [280, 301]
    colors = [primary_colors[0], primary_colors[1]]
    
    bars = plt.bar(disorders, cases_millions, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    plt.title('Global Mental Health Disorders\n(Millions of people)', 
              fontsize=16, fontweight='bold', color=primary_colors[0], pad=20)
    plt.ylabel('Millions of people', fontsize=14, color=primary_colors[1])
    plt.xlabel('Type of Disorder', fontsize=14, color=primary_colors[1])
    
    # Add values on bars
    for i, v in enumerate(cases_millions):
        plt.text(i, v + 5, f'{v}M', ha='center', fontweight='bold', 
                fontsize=12, color=primary_colors[0])
    
    # Style the plot
    plt.gca().set_facecolor('#F8F9FA')
    plt.grid(True, alpha=0.3, color=primary_colors[2])
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('global_disorders.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Graph 2: Mental Disorders Prevalence in Mexico
def create_mexico_prevalence_chart():
    plt.figure(figsize=(10, 8))
    population_mx = ['With Mental\nDisorder', 'Without Mental\nDisorder']
    percentages_mx = [15, 85]
    colors_mx = [primary_colors[1], primary_colors[3]]
    
    wedges, texts, autotexts = plt.pie(percentages_mx, labels=population_mx, 
                                       colors=colors_mx, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 12})
    
    # Style the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_color(primary_colors[0])
        text.set_fontweight('bold')
    
    plt.title('Mental Disorder Prevalence in Mexico', 
              fontsize=16, fontweight='bold', color=primary_colors[0], pad=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('mexico_prevalence.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Graph 3: Professional Care Access in Mexico
def create_care_access_chart():
    plt.figure(figsize=(10, 8))
    care = ['Receive Adequate\nProfessional Care', 'Do Not Receive\nAdequate Care']
    percentages_care = [20, 80]
    colors_care = [primary_colors[2], accent_colors[4]]
    
    wedges, texts, autotexts = plt.pie(percentages_care, labels=care, 
                                       colors=colors_care, autopct='%1.1f%%', 
                                       startangle=90, textprops={'fontsize': 12})
    
    # Style the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_color(primary_colors[0])
        text.set_fontweight('bold')
    
    plt.title('Professional Care Access in Mexico\n(Among those with mental disorders)', 
              fontsize=16, fontweight='bold', color=primary_colors[0], pad=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('care_access.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Graph 4: Global vs Mexico Comparison
def create_comparison_chart():
    plt.figure(figsize=(12, 8))
    
    x = np.arange(3)
    width = 0.35
    
    global_data = [280, 301, 0]  # Millions
    mexico_data_millions = [0, 0, 19.5]  # 15% of ~130 million approx
    
    bars1 = plt.bar(x - width/2, global_data, width, label='Global (Millions)', 
                   color=primary_colors[0], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = plt.bar(x + width/2, mexico_data_millions, width, label='Mexico (Estimated Millions)', 
                   color=primary_colors[1], alpha=0.8, edgecolor='white', linewidth=2)
    
    plt.xlabel('Type of Disorder', fontsize=14, color=primary_colors[1])
    plt.ylabel('Millions of people', fontsize=14, color=primary_colors[1])
    plt.title('Global vs Mexico Comparison - Mental Disorders\n(in millions of people)', 
              fontsize=16, fontweight='bold', color=primary_colors[0], pad=20)
    plt.xticks(x, ['Depression', 'Anxiety', 'Mental Disorders\n(General)'])
    
    # Style the legend
    legend = plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(primary_colors[2])
    
    # Add values on bars
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.annotate(f'{height}M',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold',
                           color=primary_colors[0])
    
    add_values(bars1)
    add_values(bars2)
    
    # Style the plot
    plt.gca().set_facecolor('#F8F9FA')
    plt.grid(True, alpha=0.3, color=primary_colors[2])
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('global_vs_mexico.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Graph 5: Mental Health Statistics Overview
def create_statistics_overview():
    plt.figure(figsize=(14, 8))
    
    categories = ['Depression\nWorldwide', 'Anxiety\nWorldwide', 'Mental Disorders\nin Mexico', 'Receive Care\nin Mexico']
    values = [280, 301, 15, 20]  # First two in millions, last two in percentages
    colors = [primary_colors[0], primary_colors[1], primary_colors[2], accent_colors[3]]
    
    bars = plt.bar(categories, values, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    plt.title('Mental Health Statistics Overview', 
              fontsize=18, fontweight='bold', color=primary_colors[0], pad=20)
    plt.ylabel('Millions / Percentage', fontsize=14, color=primary_colors[1])
    
    # Add values on bars with appropriate units
    units = ['M', 'M', '%', '%']
    for bar, value, unit in zip(bars, values, units):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value}{unit}', ha='center', va='bottom', fontweight='bold',
                fontsize=12, color=primary_colors[0])
    
    # Style the plot
    plt.gca().set_facecolor('#F8F9FA')
    plt.grid(True, alpha=0.3, color=primary_colors[2])
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # Add footer note
    plt.figtext(0.5, 0.02, 'Sources: WHO (2021), Mexican Ministry of Health (2020)', 
                ha='center', fontsize=10, style='italic', color=primary_colors[2])
    
    # Save the figure
    plt.savefig('statistics_overview.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# Create all charts
if __name__ == "__main__":
    print("Creating Chart 1: Global Mental Health Disorders...")
    create_global_disorders_chart()
    
    print("Creating Chart 2: Mexico Prevalence...")
    create_mexico_prevalence_chart()
    
    print("Creating Chart 3: Care Access...")
    create_care_access_chart()
    
    print("Creating Chart 4: Global vs Mexico Comparison...")
    create_comparison_chart()
    
    print("Creating Chart 5: Statistics Overview...")
    create_statistics_overview()
    
    print("All charts have been created and saved!")