# Pageviews badge injector for Chirpy theme
# Place this file in _plugins/pageviews_injector.rb
# This will automatically add hit counters to all posts without modifying theme files

Jekyll::Hooks.register :posts, :post_render do |post|
  # Get the post URL
  post_url = post.url

  # Create the badge HTML
  badge_html = <<~HTML
    <!-- pageviews -->
          <span>
            <img style="height: 1.2em; vertical-align: baseline; opacity: 0.8; transform: translateY(4px);"
                 src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fjybsuper.github.io%2F#{post_url}%2F&label=&icon=eye-fill&color=%23666666&message=&style=flat&tz=America%2FLos_Angeles"
                 alt="View Count" />
          </span>
  HTML

  # Replace the <!-- pageviews --> comment with our badge
  # This is more reliable than looking for specific HTML elements
  post.output = post.output.gsub(/<!-- pageviews -->/, badge_html)
end
