There are some things to note about the RAPID model and the associated converter. 

Since RAPID uses the first trigger point as a reference, and PLAsTiCC can set the `detected` column
to 1 on the first frame the data might be cut off. This shouldn't negatively impact performance
for short-lived light curves (like supernovae)

Also, there is some hacky stuff going on in the train function because it expects a function
to return data (which is dumb and out of scope for it. I may rewrite to use the internal functions
at some point depending on how bad it gets).