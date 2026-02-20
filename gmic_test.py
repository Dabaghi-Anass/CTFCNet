import gmic

# On Linux a window shall open-up and display a blurred earth
gmic.run("sp earth blur 4 display")
# Filter a rose with bokeh effect and get the result as a gmic.ImageList
imglst = gmic.run("sp rose fx_bokeh 3,8,0,30,8,4,0.3,0.2,210,210,80,160,0.7,30,20,20,1,2,170,130,20,110,0.15,0")
# Save the image from the previous run() to a file
gmic.run("output rose_with_bokeh.png", imglst)
