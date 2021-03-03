from django import forms

class GetTweetForm(forms.Form):
    tweet = forms.CharField(label="Enter Tweet", widget=forms.Textarea)
