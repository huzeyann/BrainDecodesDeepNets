window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // TOC Navigation: Smooth scrolling and active section highlighting
    $('.toc-link').on('click', function(e) {
      e.preventDefault();
      var targetId = $(this).attr('href');
      
      // If href is "#" or empty, scroll to top
      if (targetId === '#' || targetId === '') {
        $('html, body').animate({
          scrollTop: 0
        }, 600);
        return;
      }
      
      var targetElement = $(targetId);
      
      if (targetElement.length) {
        var offset = $('.toc-navbar').outerHeight() + 20;
        $('html, body').animate({
          scrollTop: targetElement.offset().top - offset
        }, 600);
      }
    });

    // Update active TOC link based on scroll position
    function updateActiveTOCLink() {
      var scrollPos = $(window).scrollTop();
      var navbarHeight = $('.toc-navbar').outerHeight();
      var currentSection = '';
      
      // If at the very top, highlight Overview link
      if (scrollPos < navbarHeight + 50) {
        $('.toc-link').removeClass('active');
        $('.toc-link[href="#"]').addClass('active');
        return;
      }
      
      scrollPos += navbarHeight + 100;
      
      // Check all elements with IDs that are in TOC
      $('section[id], div[id]').each(function() {
        var elementTop = $(this).offset().top;
        var elementId = $(this).attr('id');
        
        // Only check if this ID is in our TOC links
        if ($('.toc-link[href="#' + elementId + '"]').length > 0) {
          if (scrollPos >= elementTop) {
            currentSection = elementId;
          }
        }
      });
      
      // Update active class
      $('.toc-link').removeClass('active');
      if (currentSection) {
        $('.toc-link[href="#' + currentSection + '"]').addClass('active');
      }
    }

    // Update on scroll
    $(window).on('scroll', updateActiveTOCLink);
    
    // Update on page load
    updateActiveTOCLink();

    var videoOptions = {
			slidesToScroll: 1,
			slidesToShow: 4,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 3000,
    }

    var baselineOptions = {
			slidesToScroll: 1,
			slidesToShow: 2,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize video carousel with 4 slides
    setTimeout(function() {
      var videoCarouselEl = document.getElementById('video-carousel');
      if (videoCarouselEl) {
        var videoCarousel = bulmaCarousel.attach('#video-carousel', videoOptions);
        console.log('Video carousel initialized:', videoCarousel);
      }
    }, 100);
    
    // Initialize baseline carousel with 2 slides
    setTimeout(function() {
      var baselineCarouselEl = document.getElementById('baseline-carousel');
      if (baselineCarouselEl) {
        var baselineCarousel = bulmaCarousel.attach('#baseline-carousel', baselineOptions);
        console.log('Baseline carousel initialized:', baselineCarousel);
      }
    }, 200);

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
