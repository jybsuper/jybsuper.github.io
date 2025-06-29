# Pageviews badge injector for Chirpy theme
# Place this file in _plugins/pageviews_injector.rb
# This will automatically add hit counters to all posts without modifying theme files

Jekyll::Hooks.register :posts, :post_render do |post|
  # Create the badge HTML (without extra indentation)
  badge_html = <<~HTML.strip
    <span>
      <img style="height: 1.2em; vertical-align: baseline; opacity: 0.8; transform: translateY(4px);"
           src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fjybsuper.github.io#{post.url}&label=&icon=eye-fill&color=%23666666&message=&style=flat&tz=America%2FLos_Angeles"
           alt="View Count" />
    </span>
  HTML

  # Insert before the readtime span
  # This matches: <span class="readtime"...>
  post.output = post.output.gsub(
    /(<span\s+class="readtime"[^>]*>)/,
    "#{badge_html} \\1"
  )
end
