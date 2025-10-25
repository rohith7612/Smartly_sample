from django.contrib import admin
from .models import (
    Document,
    ProcessedResult,
    YouTubeVideo,
    YouTubeProcessedResult,
    ChatSession,
    ChatMessage,
)

class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "document_type", "processing_type", "uploaded_at", "user")
    list_filter = ("document_type", "processing_type", "uploaded_at", "user")
    search_fields = ("title", "user__username")
    date_hierarchy = "uploaded_at"
    list_per_page = 25

class ProcessedResultAdmin(admin.ModelAdmin):
    def short_text(self, obj):
        return (obj.result_text or "").strip()[:80]
    short_text.short_description = "Result preview"

    list_display = ("document", "short_text", "processed_at")
    list_filter = ("processed_at",)
    search_fields = ("document__title",)
    date_hierarchy = "processed_at"
    list_per_page = 25

class YouTubeVideoAdmin(admin.ModelAdmin):
    list_display = ("title", "url", "user", "processed_at")
    list_filter = ("processed_at", "user")
    search_fields = ("title", "url", "user__username")
    date_hierarchy = "processed_at"
    list_per_page = 25

class YouTubeProcessedResultAdmin(admin.ModelAdmin):
    def short_text(self, obj):
        return (obj.result_text or "").strip()[:80]
    short_text.short_description = "Result preview"

    list_display = ("youtube_video", "processing_type", "user", "short_text", "processed_at")
    list_filter = ("processing_type", "processed_at", "user")
    search_fields = ("youtube_video__title", "youtube_video__url", "user__username")
    date_hierarchy = "processed_at"
    list_per_page = 25

class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ("role", "content", "created_at")
    can_delete = False

class ChatSessionAdmin(admin.ModelAdmin):
    def document_count(self, obj):
        return obj.documents.count()
    document_count.short_description = "Documents"

    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = "Messages"

    list_display = ("title", "user", "document_count", "message_count", "created_at")
    list_filter = ("created_at", "user")
    search_fields = ("title", "user__username")
    date_hierarchy = "created_at"
    inlines = [ChatMessageInline]
    list_per_page = 25

class ChatMessageAdmin(admin.ModelAdmin):
    def short_content(self, obj):
        return (obj.content or "").strip()[:80]
    short_content.short_description = "Content"

    list_display = ("session", "role", "short_content", "created_at")
    list_filter = ("role", "created_at")
    search_fields = ("content", "session__title", "session__user__username")
    date_hierarchy = "created_at"
    list_per_page = 50

admin.site.register(Document, DocumentAdmin)
admin.site.register(ProcessedResult, ProcessedResultAdmin)
admin.site.register(YouTubeVideo, YouTubeVideoAdmin)
admin.site.register(YouTubeProcessedResult, YouTubeProcessedResultAdmin)
admin.site.register(ChatSession, ChatSessionAdmin)
admin.site.register(ChatMessage, ChatMessageAdmin)
