from django.shortcuts import render
from .forms import GetTweetForm
from django.http import HttpResponseRedirect
from core.naivebayes.naive_bayes_sentiment import naive_bayes_predict

# Create your views here.
def home_view(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = GetTweetForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            tweet = form.cleaned_data['tweet']
            print(tweet)
            p = naive_bayes_predict(tweet)
            if p >= 0.5:
                sentiment = "Positive"
            elif p <= -0.5:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            print("score: ", p)
            print("sentiment: ", sentiment)
            # redirect to a new URL:
            form = GetTweetForm()
            return render(request, 'core/home.html', {'form': form, 'tweet': tweet, 'sentiment': sentiment})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = GetTweetForm()

    return render(request, 'core/home.html', {'form': form})

def index_view(request):
    return render(request, 'core/index.html')
