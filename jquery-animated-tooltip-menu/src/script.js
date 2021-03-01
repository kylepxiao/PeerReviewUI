$(document).ready(function(){
  

$('.tooltip li').find('ul').addClass('level-2');    $('.level-2 li').find('ul').removeClass('level-2').addClass('level-3'); 
 $('.level-2 li').width('200px');

     
 $('.tooltip li span').hover( function(){
     $(this).next('.level-2').show();  
},
    function(){
    $(this).next('.level-2').fadeOut("fast");  
 });
  $('.level-2').mouseenter(function() {
  $(this).stop().css('opacity', '1');  
});


$('.level-2').mouseleave(function() {
    $(this).hide(); 
$(this).find('.level-3').slideUp(); 
$('li').removeClass('push-two push-one push-three');

$(this).find('.fa-caret-down').removeClass('flip'); 
$(this).find('.fa-caret-down').hide();

   });
   
 $('.level-2 li').each(function(){
        if ($(this).find('ul').length == 0)
        {
      $(this).addClass('solo');
        }
        if ($(this).find('ul').length != 0)
        {
$(this).prepend('<i class="fa fa-caret-down"></i>');
        }
 $(this).find('i').hide();  
});
    

  
$('.level-2 li span').hover( function(){
     
       $(this).parent().find('.fa-caret-down').toggle();  
  if ($(this).parent().find('.level-3').is(':visible')) {
   
$(this).parent().find('.fa-caret-down').show();
}
});

$('.level-2 li span').click(function(){
$(this).parent().find('.fa-caret-down').toggleClass('flip'); 

  $(this).next('.level-3').slideToggle("fast");

if ($(this).next('ul').find('li').length == 1 ) {
$(this).parent().next().toggleClass('push-one');
}

if ($(this).next('ul').find('li').length == 2 ) {
$(this).parent().next().toggleClass('push-two');
}

if ($(this).next('ul').find('li').length == 3 ) {
$(this).parent().next().toggleClass('push-three');
}

 });
 
 });