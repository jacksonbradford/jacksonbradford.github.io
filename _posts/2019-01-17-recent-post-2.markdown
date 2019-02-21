---
layout: post
title:  "Recent #2"
date:   2019-01-17 15:31:21 -0600
categories: jekyll update
image:	"/assets/placeholder_header.png"
---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.

# coding: utf8
@decorator(param=1)
def f(x):
    """ Syntax Highlighting Demo
	@param x Parameter"""
    s = ("Test", 2+3, {'a': 'b'}, x)   #Comment
    print s[0].lower()

class Foo:
    def __init__(self):
        byte_string = 'newline:\n also newline:\x0a'
        text_string = u"Cyrillic Я is \u042f. Oops: \u042g"
        self.makeSense(whatever=1)

    def makeSense(self, whatever):
        self.sense = whatever

x = len('abc')
print(f.__doc__)

require "test"
CONSTANT = 777

# Sample comment

class Module::Class
  include Testcase

  render :action => 'foo'
  def foo(parameter)
    @parameter = parameter
  end

  local_var = eval <<-"FOO";\
  printIndex "Hello world!"
  And now this is heredoc!
  printIndex "Hello world again!"
  FOO
  foo("#{$GLOBAL_TIME >> $`} is \Z sample \"string\"" * 777);
  if ($1 =~ /sample regular expression/ni) then
  begin
    puts %W(sample words), CONSTANT, :fooo;
    do_something :action => "action"
  end
  expect{counter[0]}.to_be eq 1
  1.upto(@@n) do |index| printIndex 'Hello' + index end
  \\\\\\\\\
  end
end
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
