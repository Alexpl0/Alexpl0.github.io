// This file contains the JavaScript functionality for the presentation.
// It handles slide navigation, progress bar updates, and keyboard controls.

let currentSlide = 1;
// Updated total number of slides to reflect the final version
let totalSlides = 24; 

function showSlide(n) {
    // Hide all slides
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => slide.style.display = 'none');
    
    // Show current slide
    const current = document.querySelector(`[data-slide="${n}"]`);
    if (current) {
        current.style.display = 'flex'; // Use flex to maintain layout
        // Trigger animation
        current.style.animation = 'none';
        current.offsetHeight; /* trigger reflow */
        current.style.animation = null; 
    }
    
    // Update counter
    document.getElementById('slideCounter').textContent = `${n} / ${totalSlides}`;
    
    // Update progress bar
    const progress = (n / totalSlides) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        showSlide(currentSlide);
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        showSlide(currentSlide);
    }
}

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === ' ') {
        nextSlide();
        e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
        previousSlide();
        e.preventDefault();
    }
});

// Initialize
showSlide(1);
